"""Explicit CUDA selection and diagnostics without changing PPO hyperparameters."""
import argparse
import torch


def resolve_training_device(requested='auto'):
    requested = str(requested).lower()
    if requested == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested == 'cpu':
        return requested
    if requested == 'cuda' or requested.startswith('cuda:'):
        device = torch.device(requested)
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but PyTorch cannot access it. '
                               'Run bash scripts/setup.sh --reinstall-torch and check the NVIDIA driver.')
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(f'CUDA device index unavailable: {requested}')
        return requested
    raise ValueError('Training device must be auto, cpu, cuda or cuda:N.')


def report_training_device(requested, selected):
    print(f'PPO device: requested={requested}, selected={selected}', flush=True)
    print(f'PyTorch: {torch.__version__} | CUDA build: {torch.version.cuda} | '
          f'CUDA available: {torch.cuda.is_available()}', flush=True)
    if selected.startswith('cuda'):
        index = torch.device(selected).index or 0
        print(f'GPU: {torch.cuda.get_device_name(index)}', flush=True)
    elif str(requested).lower() == 'auto':
        print('CUDA unavailable: using CPU. This does not change the PPO architecture.', flush=True)
    print('The NumPy vehicle simulation runs on CPU; the selected device runs the PPO network.', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--check', action='store_true', help='Also execute a small operation on the selected device')
    args = parser.parse_args(argv)
    selected = resolve_training_device(args.device)
    report_training_device(args.device, selected)
    if args.check:
        value = torch.ones((16, 16), device=selected)
        assert (value @ value).sum().item() == 4096
        print(f'Device operation verified on {selected}.', flush=True)


if __name__ == '__main__':
    main()
