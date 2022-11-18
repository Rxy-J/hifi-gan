import torch

def stft(x, fft_size, hop_size, win_size, window):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x: Input signal tensor (B, T).

    Returns:
        Tensor: Magnitude spectrogram (B, T, fft_size // 2 + 1).

    """
    x_stft = torch.stft(x, fft_size, hop_size, win_size, window)
    real = x_stft[..., 0]
    imag = x_stft[..., 1]
    outputs = torch.clamp(real**2 + imag**2, min=1e-7).transpose(2, 1)
    outputs = torch.sqrt(outputs)

    return outputs


class SpectralConvergence(torch.nn.Module):

    def __init__(self):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergence, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        x = torch.norm(targets_mag - predicts_mag, p='fro')
        y = torch.norm(targets_mag, p='fro')

        return x / y


class LogSTFTMagnitude(torch.nn.Module):

    def __init__(self):
        super(LogSTFTMagnitude, self).__init__()

    def forward(self, predicts_mag, targets_mag):
        log_predicts_mag = torch.log(predicts_mag)
        log_targets_mag = torch.log(targets_mag)
        outputs = torch.nn.functional.l1_loss(log_predicts_mag, log_targets_mag)

        return outputs


class STFTLoss(torch.nn.Module):

    def __init__(self, fft_size=1024, hop_size=120, win_size=600):
        super(STFTLoss, self).__init__()

        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_size = win_size
        self.register_buffer('window', torch.hann_window(win_size))
        self.sc_loss = SpectralConvergence()
        self.mag = LogSTFTMagnitude()

    def forward(self, predicts, targets):
        """
        Args:
            x: predicted signal (B, T).
            y: truth signal (B, T).

        Returns:
            Tensor: STFT loss values.
        """
        predicts_mag = stft(predicts, self.fft_size, self.hop_size,
                            self.win_size, self.window)
        targets_mag = stft(targets, self.fft_size, self.hop_size, self.win_size,
                           self.window)

        sc_loss = self.sc_loss(predicts_mag, targets_mag)
        mag_loss = self.mag(predicts_mag, targets_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    def __init__(self,
                 fft_sizes=[128, 256, 256],
                 win_sizes=[80, 160, 200],
                 hop_sizes=[20, 40, 50]):
        super(MultiResolutionSTFTLoss, self).__init__()

        self.loss_layers = torch.nn.ModuleList()
        for (fft_size, win_size, hop_size) in zip(fft_sizes, win_sizes, hop_sizes):
            self.loss_layers.append(STFTLoss(fft_size, hop_size, win_size))

    def forward(self, fake_signals, true_signals):
        sc_losses = []
        mag_losses = []
        for layer in self.loss_layers:
            sc_loss, mag_loss = layer(fake_signals, true_signals)
            sc_losses.append(sc_loss)
            mag_losses.append(mag_loss)

        sc_loss = sum(sc_losses) / len(sc_losses)
        mag_loss = sum(mag_losses) / len(mag_losses)

        return sc_loss, mag_loss


def feature_loss(fmap_r, fmap_g):
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            loss += torch.mean(torch.abs(rl - gl))

    return loss*2

def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        r_loss = torch.mean((1-dr)**2)
        g_loss = torch.mean(dg**2)
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses

def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        l = torch.mean((1-dg)**2)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses
