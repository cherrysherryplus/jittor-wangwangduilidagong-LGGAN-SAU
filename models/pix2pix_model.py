"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import jittor as jt
import copy
import models.networks as networks
from models.networks import loss
import util.util as util


class Pix2PixModel(jt.nn.Module):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        networks.modify_commandline_options(parser, is_train)
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt

        self.netG, self.netD, self.netE = self.initialize_networks(opt)

        # set loss functions
        if opt.isTrain:
            self.criterionGAN = networks.GANLoss(opt.gan_mode, opt=self.opt)
            self.criterionFeat = jt.nn.L1Loss()
            self.criterionL1 = jt.nn.L1Loss()
            self.criterionCE = jt.nn.cross_entropy_loss
            if not opt.no_vgg_loss:
                self.criterionVGG = networks.VGGLoss(self.opt.gpu_ids)
            if not opt.no_tv_loss:
                self.criterionTV = loss.TVLoss()
            if opt.use_vae:
                self.KLDLoss = networks.KLDLoss()

    # Entry point for all calls involving forward pass
    # of deep networks. We used this approach since DataParallel module
    # can't parallelize custom functions, we branch to different
    # routines based on |mode|.
    def execute(self, data, mode):
        input_semantics, real_image = self.preprocess_input(data) # print(input_semantics.size()) [1, 36, 256, 512] print(real_image.size()) [1, 3, 256, 512]
        # print(label_map.size()) [1, 1, 256, 512]

        if mode == 'generator':
            g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
            label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
            label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, result_0, result_1, result_2, result_3, result_4, \
            result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
            result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
            result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , \
             feature_score, target, index, attention_global, attention_local = self.compute_generator_loss(input_semantics, real_image)
            return g_loss, generated, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ,feature_score, target, index, attention_global, attention_local
        elif mode == 'discriminator':
            d_loss = self.compute_discriminator_loss(input_semantics, real_image)
            return d_loss
        elif mode == 'encode_only':
            z, mu, logvar = self.encode_z(real_image)
            return mu, logvar
        elif mode == 'inference':
            with jt.no_grad():
                fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , feature_score, target, index,  attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)
            return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ,feature_score, target, index,  attention_global, attention_local
        else:
            raise ValueError("|mode| is invalid")

    def create_optimizers(self, opt):
        G_params = list(self.netG.parameters())
        if opt.use_vae:
            G_params += list(self.netE.parameters())
        if opt.isTrain:
            D_params = list(self.netD.parameters())

        if opt.no_TTUR:
            beta1, beta2 = opt.beta1, opt.beta2
            G_lr, D_lr = opt.lr, opt.lr
        else:
            beta1, beta2 = 0, 0.9
            G_lr, D_lr = opt.lr / 2, opt.lr * 2

        optimizer_G = jt.optim.Adam(G_params, lr=G_lr, betas=(beta1, beta2))
        optimizer_D = jt.optim.Adam(D_params, lr=D_lr, betas=(beta1, beta2))

        return optimizer_G, optimizer_D

    def save(self, epoch):
        util.save_network(self.netG, 'G', epoch, self.opt)
        util.save_network(self.netD, 'D', epoch, self.opt)
        if self.opt.use_vae:
            util.save_network(self.netE, 'E', epoch, self.opt)

    ############################################################################
    # Private helper methods
    ############################################################################

    def initialize_networks(self, opt):
        netG = networks.define_G(opt)
        netD = networks.define_D(opt) if opt.isTrain else None
        netE = networks.define_E(opt) if opt.use_vae else None

        if not opt.isTrain or opt.continue_train:
            netG = util.load_network(netG, 'G', opt.which_epoch, opt)
            if opt.isTrain:
                netD = util.load_network(netD, 'D', opt.which_epoch, opt)
            if opt.use_vae:
                netE = util.load_network(netE, 'E', opt.which_epoch, opt)

        return netG, netD, netE
        

    # preprocess the input, such as moving the tensors to GPUs and
    # transforming the label map to one-hot encoding
    # |data|: dictionary of the input data

    def preprocess_input(self, data):
        # move to GPU and change data types
        data['label'] = data['label'].long()
        if self.use_gpu():
            data['label'] = data['label']
            # assign None
            data['instance'] = None
            data['image'] = data['image']

        # create one-hot label map
        label_map = data['label'] # print(label_map.size()) [1, 1, 256, 512]
        bs, _, h, w = label_map.size()
        nc = self.opt.label_nc + 1 if self.opt.contain_dontcare_label \
            else self.opt.label_nc
        input_label = jt.zeros(shape=(bs, nc, h, w)) # print(input_label.size()) [1, 35, 256, 512]
        input_semantics = input_label.scatter_(1, label_map, jt.array(1.0)) # print(input_semantics.size()) [1, 35, 256, 512]

        # concatenate instance map if it exists
        if not self.opt.no_instance:
            inst_map = data['instance']
            instance_edge_map = self.get_edges(inst_map)
            input_semantics = jt.concat((input_semantics, instance_edge_map), dim=1)

        return input_semantics, data['image']

    def compute_generator_loss(self, input_semantics, real_image):
        G_losses = {}

        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ,feature_score, target, index, attention_global, attention_local, KLD_loss = self.generate_fake(input_semantics, real_image, compute_kld_loss=self.opt.use_vae)

        if self.opt.use_vae:
            G_losses['KLD'] = KLD_loss

        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_local, pred_real_local = self.discriminate(input_semantics, result_local, real_image)

        G_losses['GAN'] = self.criterionGAN(pred_fake_generated, True, for_discriminator=False) + \
                          self.criterionGAN(pred_fake_global, True, for_discriminator=False) + \
                          self.criterionGAN(pred_fake_local, True, for_discriminator=False)

        if not self.opt.no_tv_loss:
            # result_1 result_2等后续可以尝试加上
            G_losses['TV'] = self.criterionTV(fake_image) + \
                             self.criterionTV(result_global) + \
                             self.criterionTV(result_local)
            G_losses['TV'] *= self.opt.lambda_tv

        if not self.opt.no_ganFeat_loss:
            num_D = len(pred_fake_generated) # print(num_D) 2

            GAN_Feat_loss = jt.float32(0)
            for i in range(num_D):  # for each discriminator, last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake_generated[i]) - 1
                for j in range(num_intermediate_outputs):  # for each layer output
                    unweighted_loss = self.criterionFeat(pred_fake_generated[i][j], pred_real_generated[i][j].detach()) + \
                                      self.criterionFeat(pred_fake_global[i][j], pred_real_global[i][j].detach()) + \
                                      self.criterionFeat(pred_fake_local[i][j], pred_real_local[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.opt.lambda_feat / num_D
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.opt.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, real_image) * self.opt.lambda_vgg + \
                              self.criterionVGG(result_global, real_image) * self.opt.lambda_vgg + \
                              self.criterionVGG(result_local, real_image) * self.opt.lambda_vgg

        if not self.opt.no_l1_loss:
            G_losses['L1'] = self.criterionL1(fake_image, real_image) * self.opt.lambda_l1 + \
                              self.criterionL1(result_global, real_image) * self.opt.lambda_l1 + \
                              self.criterionL1(result_local, real_image) * self.opt.lambda_l1

        if not self.opt.no_class_loss:
            # 下面两行是针对jittor添加的。jittor的ce loss在计算时，c代表的是类别数
            # output不能输入[b,c,h]的，只能输入[b,c]和[b,c,h,w]的。
            # target不能输入[b,h]的，只能输入[b,]和[b,h,w]的，
            # 如果target为类别索引，取值只能在[0,c-1]范围内，否则类别概率则在[0,1]范围内。
            # 具体可以看torch.nn.CrossEntropyLoss的文档和jt.nn.cross_entropy_loss的源码
            target_ = target.repeat([self.opt.batchSize, 1]).unsqueeze(-1)
            feature_score_ = feature_score.view((self.opt.batchSize, self.opt.semantic_nc, -1, 1))
            # 下面这行代码会出问题。不加squeeze(-1)会导致class损失计算结果比正常高几个数量级
            G_losses['class'] = jt.sum(self.criterionCE(feature_score_, target_, reduction='none').squeeze(-1) \
                                * index) / jt.sum(index) * self.opt.lambda_class

        # TODO: local pixel loss
        if not self.opt.no_l1_local_loss:
            real_image = real_image
            real_0 = real_image * label_3_0
            real_1 = real_image * label_3_1
            real_2 = real_image * label_3_2
            real_3 = real_image * label_3_3
            real_4 = real_image * label_3_4
            real_5 = real_image * label_3_5
            real_6 = real_image * label_3_6
            real_7 = real_image * label_3_7
            real_8 = real_image * label_3_8
            real_9 = real_image * label_3_9
            real_10 = real_image * label_3_10
            real_11 = real_image * label_3_11
            real_12 = real_image * label_3_12
            real_13 = real_image * label_3_13
            real_14 = real_image * label_3_14
            real_15 = real_image * label_3_15
            real_16 = real_image * label_3_16
            real_17 = real_image * label_3_17
            real_18 = real_image * label_3_18
            real_19 = real_image * label_3_19
            real_20 = real_image * label_3_20
            real_21 = real_image * label_3_21
            real_22 = real_image * label_3_22
            real_23 = real_image * label_3_23
            real_24 = real_image * label_3_24
            real_25 = real_image * label_3_25
            real_26 = real_image * label_3_26
            real_27 = real_image * label_3_27
            real_28 = real_image * label_3_28

            G_losses['L1_Local'] = self.opt.lambda_l1 * ( self.criterionL1(result_0, real_0) + \
                              self.criterionL1(result_1, real_1) + self.criterionL1(result_2, real_2) + \
                              self.criterionL1(result_3, real_3) + self.criterionL1(result_4, real_4) + \
                              self.criterionL1(result_5, real_5) + self.criterionL1(result_6, real_6) + \
                              self.criterionL1(result_7, real_7) + self.criterionL1(result_8, real_8) + \
                              self.criterionL1(result_9, real_9) + self.criterionL1(result_10, real_10) + \
                              self.criterionL1(result_11, real_11) + self.criterionL1(result_12, real_12) + \
                              self.criterionL1(result_13, real_13) + self.criterionL1(result_14, real_14) + \
                              self.criterionL1(result_15, real_15) + self.criterionL1(result_16, real_16) + \
                              self.criterionL1(result_17, real_17) + self.criterionL1(result_18, real_18) + \
                              self.criterionL1(result_19, real_19) + self.criterionL1(result_20, real_20) + \
                              self.criterionL1(result_21, real_21) + self.criterionL1(result_22, real_22) + \
                              self.criterionL1(result_23, real_23) + self.criterionL1(result_24, real_24) + \
                              self.criterionL1(result_25, real_25) + self.criterionL1(result_26, real_26) + \
                              self.criterionL1(result_27, real_27) + self.criterionL1(result_28, real_28) )

        return G_losses, fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
                label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
                label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  \
                result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
                result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
                result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 , \
                feature_score, target, index, \
                attention_global, attention_local


    def compute_discriminator_loss(self, input_semantics, real_image):
        D_losses = {}

        with jt.no_grad():
            fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28, \
               result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ,  \
               feature_score, target, valid_index, attention_global, attention_local, _ = self.generate_fake(input_semantics, real_image)

        pred_fake_generated, pred_real_generated = self.discriminate(input_semantics, fake_image, real_image)
        pred_fake_global, pred_real_global = self.discriminate(input_semantics, result_global, real_image)
        pred_fake_lcoal, pred_real_local = self.discriminate(input_semantics, result_local, real_image)

        D_losses['D_Fake'] = self.criterionGAN(pred_fake_generated, False, for_discriminator=True) + \
                             self.criterionGAN(pred_fake_global, False, for_discriminator=True) + \
                             self.criterionGAN(pred_fake_lcoal, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real_generated, True, for_discriminator=True) + \
                             self.criterionGAN(pred_real_global, True, for_discriminator=True) + \
                             self.criterionGAN(pred_real_local, True, for_discriminator=True)

        return D_losses

    def encode_z(self, real_image):
        mu, logvar = self.netE(real_image)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def generate_fake(self, input_semantics, real_image, compute_kld_loss=False):
        z = None
        KLD_loss = None
        if self.opt.use_vae:
            z, mu, logvar = self.encode_z(real_image)
            if compute_kld_loss:
                KLD_loss = self.KLDLoss(mu, logvar) * self.opt.lambda_kld

        fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
        label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
        label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  \
        result_0, result_1, result_2, result_3, result_4, result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , result_11 ,result_12 , result_13 ,\
        result_14 , result_15 , result_16 , result_17, result_18 , result_19 , result_20, result_21 , result_22 , result_23 , result_24 , result_25 , result_26, \
        result_27 , result_28, \
        feature_score, target, index, \
        attention_global, attention_local = self.netG(input_semantics, z=z)

        assert (not compute_kld_loss) or self.opt.use_vae, \
            "You cannot compute KLD loss if opt.use_vae == False"

        return fake_image, result_global, result_local, label_3_0, label_3_1, label_3_2, label_3_3, label_3_4, label_3_5, label_3_6, label_3_7, label_3_8, \
               label_3_9, label_3_10, label_3_11, label_3_12, label_3_13, label_3_14, label_3_15, label_3_16, label_3_17, label_3_18, label_3_19, label_3_20, \
               label_3_21,label_3_22, label_3_23, label_3_24, label_3_25, label_3_26, label_3_27, label_3_28,  result_0, result_1, result_2, result_3, result_4,result_5 ,result_6 ,result_7 , result_8 , result_9 , result_10 , \
               result_11 ,result_12 , result_13 , result_14 , result_15 , result_16 , result_17 , result_18 , result_19 , result_20, \
               result_21 , result_22 , result_23 , result_24 , result_25 , result_26 , result_27 , result_28 ,feature_score, target,index, attention_global, attention_local, KLD_loss

    # Given fake and real image, return the prediction of discriminator
    # for each fake and real image.

    def discriminate(self, input_semantics, fake_image, real_image):
        fake_concat = jt.concat([input_semantics, fake_image], dim=1)
        real_concat = jt.concat([input_semantics, real_image], dim=1)

        # In Batch Normalization, the fake and real images are
        # recommended to be in the same batch to avoid disparate
        # statistics in fake and real images.
        # So both fake and real images are fed to D all at once.
        fake_and_real = jt.concat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)

        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    # Take the prediction of fake and real images from the combined batch
    def divide_pred(self, pred):
        # the prediction contains the intermediate outputs of multiscale GAN,
        # so it's usually a list
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    def get_edges(self, t):
        edge = jt.zeros(t.size())
        edge = jt.bool(edge)
        edge[:, :, :, 1:] = edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1])
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :])
        return edge.float()

    def reparameterize(self, mu, logvar):
        std = jt.exp(0.5 * logvar)
        eps = jt.randn_like(std)
        return eps.multiply(std) + mu

    def use_gpu(self):
        return len(self.opt.gpu_ids) > 0

