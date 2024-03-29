from __future__ import print_function
import argparse
import random
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import os
import json
from collections import deque
from tensorboardX import SummaryWriter

from inception_score import inception_score
from fid import fid_score

import models.dcgan as dcgan
import models.mlp as mlp
import models.resnet as resnet


def main(opt, reporter=None):
    writer = SummaryWriter()

    with open(writer.file_writer.get_logdir() + '/args.json', 'w') as f:
        json.dump(opt, f)

    if opt['experiment'] is None:
        opt['experiment'] = 'samples'
    os.system('mkdir {0}'.format(opt['experiment']))

    opt['manualSeed'] = random.randint(1, 10000)  # fix seed
    print("Random Seed: ", opt['manualSeed'])
    random.seed(opt['manualSeed'])
    torch.manual_seed(opt['manualSeed'])

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt['cuda']:
        print("WARNING: You have a CUDA device,"
              "so you should probably run with --cuda")

    if opt['dataset'] in ['imagenet', 'folder', 'lfw']:
        # folder dataset
        dataset = dset.ImageFolder(root=opt['dataroot'],
                                   transform=transforms.Compose([
                                       transforms.Scale(opt['imageSize']),
                                       transforms.CenterCrop(opt['imageSize']),
                                       transforms.ToTensor(),
                                       transforms.Normalize((0.5, 0.5, 0.5),
                                                            (0.5, 0.5, 0.5)),
                                   ]))
    elif opt['dataset'] == 'lsun':
        dataset = dset.LSUN(root=opt['dataroot'], classes=['bedroom_train'],
                            transform=transforms.Compose([
                                transforms.Scale(opt['imageSize']),
                                transforms.CenterCrop(opt['imageSize']),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5),
                                                     (0.5, 0.5, 0.5)),
                            ]))
    elif opt['dataset'] == 'cifar10':
        dataset = dset.CIFAR10(root=opt['dataroot'], download=True,
                               transform=transforms.Compose([
                                   transforms.Scale(opt['imageSize']),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5),
                                                        (0.5, 0.5, 0.5)),
                               ]))

    assert dataset
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=opt['batchSize'],
                                             shuffle=True,
                                             num_workers=int(opt['workers']))

    ngpu = int(opt['ngpu'])
    nz = int(opt['nz'])
    ngf = int(opt['ngf'])
    ndf = int(opt['ndf'])
    nc = int(opt['nc'])
    n_extra_layers = int(opt['n_extra_layers'])

    # custom weights initialization called on netG and netD
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data.normal_(0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    if opt['noBN']:
        netG = dcgan.DCGAN_G_nobn(opt['imageSize'], nz, nc, ngf, ngpu,
                                  n_extra_layers)
    elif opt['type'] == 'mlp':
        netG = mlp.MLP_G(opt['imageSize'], nz, nc, ngf, ngpu)
    elif opt['type'] == 'resnet':
        netG = resnet.Generator(nz)
    else:
        netG = dcgan.DCGAN_G(opt['imageSize'], nz, nc, ngf, ngpu,
                             n_extra_layers)

    netG.apply(weights_init)
    print(netG)

    if opt['type'] == 'mlp':
        netD = mlp.MLP_D(opt['imageSize'], nz, nc, ndf, ngpu)
    elif opt['type'] == 'resnet':
        netD = resnet.Discriminator(nz)
    else:
        netD = dcgan.DCGAN_D(opt['imageSize'], nz, nc, ndf, ngpu, n_extra_layers)
        netD.apply(weights_init)

    print(netD)

    inc_noise = torch.utils.data.TensorDataset(torch.randn(50000, nz, 1,
                                                           1).cuda())
    inc_noise_dloader = torch.utils.data.DataLoader(inc_noise,
                                                    batch_size=opt['batchSize'])

    input = torch.FloatTensor(opt['batchSize'], 3, opt['imageSize'],
                              opt['imageSize'])
    noise = torch.FloatTensor(opt['batchSize'], nz, 1, 1)
    fixed_noise = torch.FloatTensor(opt['batchSize'], nz, 1, 1).normal_(0, 1)
    one = torch.FloatTensor([1])
    mone = one * -1

    if opt['cuda']:
        netD.cuda()
        netG.cuda()
        input = input.cuda()
        one, mone = one.cuda(), mone.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

    # setup optimizer
    if opt['adam']:
        optimizerD = optim.Adam(netD.parameters(), lr=opt['lrD'],
                                betas=(opt['beta1'], opt['beta2']))
        optimizerG = optim.Adam(netG.parameters(), lr=opt['lrG'],
                                betas=(opt['beta1'], opt['beta2']))
    else:
        optimizerD = optim.RMSprop(netD.parameters(), lr=opt['lrD'])
        optimizerG = optim.RMSprop(netG.parameters(), lr=opt['lrG'])

    var_weight = 0.5
    w = torch.tensor([var_weight * (1 - var_weight)**i
                      for i in range(9, -1, -1)]).cuda()

    gen_iterations = 0
    for epoch in range(opt['niter']):
        data_iter = iter(dataloader)
        i = 0
        while i < len(dataloader):
            # l_var = opt.l_var + (gen_iterations + 1)/3000
            l_var = opt['l_var']
            ############################
            # (1) Update D network
            ###########################
            for p in netD.parameters():  # reset requires_grad
                p.requires_grad = True

            # train the discriminator Diters times
            # if gen_iterations < 25 or gen_iterations % 500 == 0:
            if gen_iterations % 500 == 0:
                Diters = 100
            else:
                Diters = opt['Diters']

            j = 0
            while j < Diters and i < len(dataloader):
                j += 1

                # enforce constraint
                if not opt['var_constraint']:
                    for p in netD.parameters():
                        p.data.clamp_(opt['clamp_lower'], opt['clamp_upper'])

                data = data_iter.next()
                i += 1

                # train with real
                real_cpu, _ = data

                netD.zero_grad()
                batch_size = real_cpu.size(0)

                if opt['cuda']:
                    real_cpu = real_cpu.cuda()
                input.resize_as_(real_cpu).copy_(real_cpu)
                inputv = Variable(input)

                out_D_real = netD(inputv)
                errD_real = out_D_real.mean(0).view(1)

                if opt['var_constraint']:
                    vm_real = out_D_real.var(0)

                # train with fake
                noise.resize_(opt['batchSize'], nz, 1, 1).normal_(0, 1)
                with torch.no_grad():
                    noisev = Variable(noise)  # totally freeze netG
                    fake = netG(noisev).data
                inputv = fake
                out_D_fake = netD(inputv)
                errD_fake = out_D_fake.mean(0).view(1)

                if opt['var_constraint']:
                    vm_fake = out_D_fake.var(0)

                errD = errD_real - errD_fake

                loss = -(
                    (errD_real - errD_fake)
                    - l_var*torch.exp(torch.sqrt(torch.log(vm_real)**2 +
                                                 torch.log(vm_fake)**2))
                )
                loss.backward()

                optimizerD.step()

                if opt['var_constraint']:
                    writer.add_scalars('train/variance', {'real': vm_real.item(),
                                                          'fake': vm_fake.item()},
                                       epoch*len(dataloader) + i)

            ############################
            # (2) Update G network
            ###########################
            for p in netD.parameters():
                p.requires_grad = False  # to avoid computation
            netG.zero_grad()
            # in case our last batch was the tail batch of the dataloader,
            # make sure we feed a full batch of noise
            noise.resize_(opt['batchSize'], nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            errG = -netD(fake).mean(0).view(1)
            errG.backward()
            optimizerG.step()
            gen_iterations += 1

            if torch.isnan(errG):
                raise ValueError("Loss is nan")

            ############################
            # Log Data
            ###########################
            print('[%d/%d][%d/%d][%d] Loss_D: %f Loss_G: %f Loss_D_real: %f'
                  ' Loss_D_fake %f' % (epoch, opt['niter'], i, len(dataloader),
                                       gen_iterations, errD.data[0], errG.data[0],
                                       errD_real.data[0], errD_fake.data[0]))
            writer.add_scalar('train/critic', -errD.item(), gen_iterations)
            if gen_iterations % (500 * 64 / opt['batchSize']) == 0:
                real_cpu = real_cpu.mul(0.5).add(0.5)
                vutils.save_image(real_cpu, f'{opt["experiment"]}/real_samples.png')
                with torch.no_grad():
                    fake = netG(Variable(fixed_noise))
                fake.data = fake.data.mul(0.5).add(0.5)
                vutils.save_image(fake.data, f'{opt["experiment"]}/'
                                  f'fake_samples_{gen_iterations:010d}.png')
                writer.add_image(
                    'train/sample',
                    fake.data.mul(255).clamp(0, 255).byte().cpu().numpy(),
                    gen_iterations
                )

            ############################
            # (3) Compute Scores
            ############################
            if gen_iterations % (500 * 64 / opt['batchSize']) == 0:
                with torch.no_grad():
                    netG.eval()
                    samples = []
                    for (x,) in inc_noise_dloader:
                        samples.append(netG(x))
                    netG.train()
                    samples = torch.cat(samples, dim=0).cpu()
                    samples = (samples - samples.mean()) / samples.std()

                score, _ = inception_score(
                    samples.numpy(),
                    cuda=True, resize=True, splits=10
                )
                writer.add_scalar('test/inception_50k', score, gen_iterations)
                # fids = fid_score(
                #     samples.permute(0, 2, 3,
                #                     1).mul(128).add(128).clamp(255).numpy(),
                #     'cifar10'
                # )
                # writer.add_scalar('test/fid_50k', fids, gen_iterations)
                if reporter:
                    reporter(inception=score, fid=0)

        # do checkpointing
        torch.save(netG.state_dict(),
                   f'{opt["experiment"]}/netG_epoch_{epoch}.pth')
        torch.save(netD.state_dict(),
                   f'{opt["experiment"]}/netD_epoch_{epoch}.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True,
                        help='cifar10 | lsun | imagenet | folder | lfw ')
    parser.add_argument('--dataroot', required=True,
                        help='path to dataset')
    parser.add_argument('--workers', type=int,
                        help='number of data loading workers', default=2)
    parser.add_argument('--batchSize', type=int, default=64,
                        help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64,
                        help='the height / width of the input image to '
                        'network')
    parser.add_argument('--nc', type=int, default=3,
                        help='input image channels')
    parser.add_argument('--nz', type=int, default=100,
                        help='size of the latent z vector')
    parser.add_argument('--ngf', type=int, default=64)
    parser.add_argument('--ndf', type=int, default=64)
    parser.add_argument('--niter', type=int, default=25,
                        help='number of epochs to train for')
    parser.add_argument('--lrD', type=float, default=0.0005,
                        help='learning rate for Critic, default=0.00005')
    parser.add_argument('--lrG', type=float, default=0.0005,
                        help='learning rate for Generator, default=0.00005')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true',
                        help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1,
                        help='number of GPUs to use')
    parser.add_argument('--netG', default='',
                        help="path to netG (to continue training)")
    parser.add_argument('--netD', default='',
                        help="path to netD (to continue training)")
    parser.add_argument('--clamp_lower', type=float, default=-0.01)
    parser.add_argument('--clamp_upper', type=float, default=0.01)
    parser.add_argument('--Diters', type=int, default=5,
                        help='number of D iters per each G iter')
    parser.add_argument('--noBN', action='store_true',
                        help='use batchnorm or not (only for DCGAN)')
    parser.add_argument('--type', default='dcgan',
                        help='dcgan | mlp | resnet')
    parser.add_argument('--adam', action='store_true',
                        help='Whether to use adam (default is rmsprop)')
    parser.add_argument('--var_constraint', action='store_true',
                        help='Whether to constrain variance instead of '
                        'lipschitz')
    opt = parser.parse_args()
    print(opt)
    main(vars(opt))
