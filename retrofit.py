import torch
import wandb

from argparse import ArgumentParser

import model


class dotdict(dict):
    """ dot.notation access to dictionary attributes """
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def load_ckpt(args):
    """Loads a trained checkpoint."""
    net = model.Model.load_from_checkpoint(args.ckpt)
    net = net.eval().requires_grad_(False).to(args.device)
    return net


def save_params(args, net):
    learned_params = {}
    for pname, p in net.named_parameters():
        if args.ft in pname:
            learned_params[pname] = p
    torch.save(learned_params, args.adapter)


def load_params(args):
    params = torch.load(args.adapter)
    keys = list(params.keys())
    rnet = model.Model(args).eval().requires_grad_(False)
    rnet_dict = rnet.state_dict()
    for pname in keys:
        rnet_dict[pname] = params[pname]
    rnet.load_state_dict(rnet_dict)
    return rnet.to(args.device)


def main():
    # Args
    parser = ArgumentParser()
    parser.add_argument('--run', type=str)
    parser.add_argument('--tmpdir', type=str)
    parser.add_argument('--savedir', type=str)
    parser.add_argument('--ckpt', type=str, default='v0')
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    print('Loading config...')
    api = wandb.Api()
    settings = api.run(args.run)
    settings.config = dotdict(settings.config)

    (user_id, project_id, run_id) = args.run.split('/')
    settings.config.ckpt = f'{args.tmpdir}/model.ckpt'
    settings.config.adapter = f'{args.savedir}/{run_id}.ckpt'
    settings.config.device = args.device

    print('Downloading ckpt...')
    run = wandb.init()
    command = f'{user_id}/{project_id}/model-{run_id}:{args.ckpt}'
    artifact = run.use_artifact(command, type='model')
    artifact.download(root=args.tmpdir)

    print('Loading ckpt...')
    net = load_ckpt(settings.config)

    print('Saving adapter and config...')
    torch.save(dict(settings.config), f'{args.savedir}/{run_id}-config')
    save_params(settings.config, net)


if __name__ == '__main__':
    main()