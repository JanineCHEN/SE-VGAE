import argparse
import torch


def get_config():
    parser = argparse.ArgumentParser(description='real-time layout design graph generation')
    parser.add_argument('--root_dir', type=str, default='logs/')
    parser.add_argument('--log_name', type=str, default='debug')
    parser.add_argument('--device', type=str, default='gpu')
    parser.add_argument('--vaetype', type=str, default='vqvae') #betavae, vae
    parser.add_argument('--gintype', type=str, default='sum') #mean, sum
    parser.add_argument('--dataset', type=str, default='FP6_basicNode') #'FP6_fullNode','FP6_basicNode','FP25_fullNode','FP25_basicNode'
    parser.add_argument('--base_lr', type=float, default=5e-7)
    parser.add_argument('--max_lr', type=float, default=5e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_epochs', type=int, default=400)
    parser.add_argument('--save_samples', type=int, default=5)
    parser.add_argument('--step_size_up', type=int, default=150)
    parser.add_argument('--resume_ep', type=int, default=199)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--lr_schedule', type=bool, default=True)
    parser.add_argument('--num_gen_samples', type=int, default=1000)
    parser.add_argument('--ZDIM', type=int, default=512)
    parser.add_argument('--svdDIM', type=int, default=8)
    parser.add_argument('--ifNED', type=bool, default=False)
    cfg = parser.parse_args()

    if not cfg.ifNED:
        cfg.log_dir = f"./outputs/{cfg.gintype}_svd{str(cfg.svdDIM)}_{cfg.vaetype}_{cfg.dataset}_{str(cfg.ZDIM)}/"
    if cfg.ifNED:
        cfg.log_dir = f"./outputs/NED_{cfg.gintype}_svd{str(cfg.svdDIM)}_{cfg.vaetype}_{cfg.dataset}_{str(cfg.ZDIM)}/"
    cfg.model_path = cfg.log_dir + '/models/'
    cfg.txt_path = cfg.log_dir + '/txt/'
    cfg.sample_path = cfg.log_dir + '/samples/'

    print(cfg)
    return cfg

if __name__ == '__main__':
    print(get_config())
