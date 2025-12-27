import torch
import torch.nn as nn
from geoopt.optim import RiemannianAdam
from geoopt.tensor import ManifoldParameter
from lib.lorentz.layers import LorentzMLR, LorentzConv2d, LorentzAvgPool2d, LorentzeLU
from lib.lorentz.manifold import CustomLorentz
import torch.nn.functional as F
import nets.batchnorm as bn


class HEEGNet(nn.Module):

    def __init__(self,
                 chunk_size: int = 124,  # Adjusted for your 124 data points
                 num_electrodes: int = 32,  # Adjusted for your 32 channels
                 F1: int = 16,  # Increased filters for higher capacity
                 F2: int = 32,  # Increased filters for higher capacity
                 D: int = 2,
                 bnorm_dispersion=bn.BatchNormDispersion.SCALAR,
                 num_classes: int = 2,
                 kernel_1: int = 32,  # Reduced kernel size for shorter 124pt signal
                 kernel_2: int = 16,
                 dropout: float = 0.25,
                 domains=[],
                 domain_adaptation=True,
                 device='cuda',
                 dtype=torch.float64,
                 lr=0.01,
                 weight_decay=1e-3):

        super(HEEGNet, self).__init__()
        self.F1 = F1
        self.F2 = F2
        self.D = D
        self.bnorm_dispersion = bnorm_dispersion
        self.chunk_size = chunk_size
        self.num_classes = num_classes
        self.num_electrodes = num_electrodes
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2
        self.dropout = dropout
        self.domain_adaptation = domain_adaptation
        self.manifold = CustomLorentz()
        self.domains = domains
        self.device_ = device
        self.dtype_ = dtype

        # Phase 1: Euclidean Feature Extraction
        # block1: Temporal and Spatial depthwise convolutions
        self.block1 = nn.Sequential(
            # Temporal Conv: kernel_1 adjusted to 32 to better fit 124 points
            nn.Conv2d(1, self.F1, (1, self.kernel_1), stride=1, padding=(0, self.kernel_1 // 2), bias=False),
            nn.BatchNorm2d(self.F1, momentum=0.01, affine=True, eps=1e-3),
            # Spatial Conv: kernel size (num_electrodes, 1) processes 32 channels
            nn.Conv2d(self.F1, self.F1 * self.D, (self.num_electrodes, 1),
                      stride=1,
                      padding=(0, 0),
                      groups=self.F1,
                      bias=False),
            nn.BatchNorm2d(self.F1 * self.D, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),  # Reduces temporal dim from 124 to ~31
            nn.Dropout(p=dropout))

        # ec1: Euclidean Refinement before hyperbolic projection
        self.ec1 = nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2),
                             stride=1, padding=(0, self.kernel_2 // 2), bias=False,
                             groups=self.F1 * self.D)

        # block2: Additional Euclidean depthwise/pointwise block for capacity
        self.block2 = nn.Sequential(
            nn.Conv2d(self.F1 * self.D, self.F1 * self.D, (1, self.kernel_2),
                      stride=1, padding=(0, self.kernel_2 // 2), bias=False, groups=self.F1 * self.D),
            nn.Conv2d(self.F1 * self.D, self.F2, 1, padding=(0, 0), groups=1, bias=False, stride=1),
            nn.BatchNorm2d(self.F2, momentum=0.01, affine=True, eps=1e-3),
            nn.ELU(),
            nn.AvgPool2d((1, 4), stride=4),  # Reduces temporal dim further for classification
            nn.Dropout(p=dropout))

        # Phase 2 & 3: Hyperbolic Bridge and Alignment
        # lc1: Pointwise Lorentz Convolution
        self.lc1 = LorentzConv2d(self.manifold, self.F1 * self.D + 1, self.F2 + 1, 1, padding=(0, 0), bias=False,
                                 stride=1)

        if domain_adaptation:
            self.bn = bn.AdaMomDomainSPDBatchNorm(self.bn_dim(), batchdim=0,
                                                  domains=self.domains,
                                                  learn_mean=False, learn_std=True,
                                                  dispersion=self.bnorm_dispersion,
                                                  eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)
        else:
            self.bn = bn.AdaMomSPDBatchNorm(self.bn_dim(), batchdim=0,
                                            dispersion=self.bnorm_dispersion,
                                            learn_mean=False, learn_std=True,
                                            eta=1., eta_test=.1, dtype=self.dtype_, device=self.device_)

        self.elu = LorentzeLU(self.manifold)
        # Lorentz AvgPool: stride reduced to 4 to prevent over-compression of the 124pt signal
        self.avpool = LorentzAvgPool2d(self.manifold, (1, 4), stride=4)

        # Phase 4: Shared Hyperbolic Classifier
        self.lmlp = LorentzMLR(self.manifold, self.feature_dim() + 1, num_classes)

    def bn_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            mock_eeg = mock_eeg.permute(0, 2, 3, 1)
            mock_eeg = mock_eeg / (mock_eeg.norm(p=2, dim=-1, keepdim=True) + 1e-8)
            mock_eeg = self.manifold.projx(F.pad(mock_eeg, pad=(1, 0)))
            mock_eeg = self.lc1(mock_eeg)
            return mock_eeg.shape

    def feature_dim(self):
        with torch.no_grad():
            mock_eeg = torch.zeros(1, 1, self.num_electrodes, self.chunk_size)
            mock_eeg = self.block1(mock_eeg)
            # Simulate the temporal path to determine final flattened size
            mock_eeg = self.ec1(mock_eeg)
            # The classification space dimensions after hyperbolic pooling
            # Using mock to calculate actual size for LorentzMLR input
            mock_eeg = self.block2(mock_eeg)
        return self.F2 * mock_eeg.shape[3]

    def forward(self, inputs, domains):
        x = inputs.unsqueeze(1)  # Input: (Batch, 1, 32, 124)
        x = self.block1(x)
        x = self.ec1(x)
        x = x.permute(0, 2, 3, 1)
        x = x / (x.norm(p=2, dim=-1, keepdim=True) + 1e-8)
        # ProjX: Bridge from Euclidean to Hyperbolic space [cite: 167, 497]
        x = self.manifold.projx(F.pad(x, pad=(1, 0)))
        x = self.lc1(x)
        # DSMDBN: Hyperbolic domain adaptation stage [cite: 171, 419]
        x = self.bn(x, domains)
        x = self.elu(x)
        x = self.avpool(x)
        # Lorentz Flattening for classification [cite: 500]
        x = self.manifold.lorentz_flatten(x)
        features = x
        # Lorentz MLR: Hyperbolic distance-based classifier [cite: 336, 499]
        x = self.lmlp(x)
        return x, features

    def configure_optimizers(self, lr=None, weight_decay=None):
        params = []
        zero_wd_params = []
        for name, param in self.named_parameters():
            if name.startswith(('lmlp', 'lc1', 'bn', 'avpool', 'elu')) and isinstance(param, ManifoldParameter):
                zero_wd_params.append(param)
            else:
                params.append(param)
        pgroups = [
            dict(params=zero_wd_params, weight_decay=0.),
            dict(params=params)
        ]
        return RiemannianAdam(pgroups, lr=lr or self.lr, weight_decay=weight_decay or self.weight_decay)

    def domainadapt_finetune(self, x, y, d, target_domains):
        if self.domain_adaptation:
            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.REFIT)
            with torch.no_grad():
                for du in d.unique():
                    self.forward(x[d == du], d[d == du])
            self.bn.set_test_stats_mode(bn.BatchNormTestStatsMode.BUFFER)