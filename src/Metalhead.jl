module Metalhead

using Flux
using Flux: outputsize, Zygote
using Functors
using BSON
using Artifacts, LazyArtifacts
using Statistics
using MLUtils
using NeuralAttentionlib

import Functors

include("utilities.jl")

# Custom Layers
include("layers/Layers.jl")
using .Layers

# CNN models
include("convnets/alexnet.jl")
include("convnets/vgg.jl")
include("convnets/inception.jl")
include("convnets/googlenet.jl")
include("convnets/resnet.jl")
include("convnets/resnext.jl")
include("convnets/densenet.jl")
include("convnets/squeezenet.jl")
include("convnets/mobilenet.jl")
include("convnets/convnext.jl")
include("convnets/convmixer.jl")

# Other models
include("other/mlpmixer.jl")
include("other/esrgan.jl")

# ViT-based models
include("vit-based/vit.jl")

include("pretrain.jl")

export  AlexNet,
        VGG, VGG11, VGG13, VGG16, VGG19,
        ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152,
        GoogLeNet, Inception3, SqueezeNet,
        DenseNet, DenseNet121, DenseNet161, DenseNet169, DenseNet201,
        ResNeXt,
<<<<<<< HEAD
        MobileNetv2, MobileNetv3,
        MLPMixer, ESRGAN
        ViT

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :GoogLeNet, :Inception3, :SqueezeNet, :DenseNet, :ResNeXt, 
          :MobileNetv2, :MobileNetv3, :MLPMixer, :ViT, :ESRGAN)
<<<<<<< HEAD
=======
=======
        MLPMixer,
=======
        MobileNetv1, MobileNetv2, MobileNetv3,
        MLPMixer, ResMLP, gMLP,
>>>>>>> aba6fb832093d88dc2d2b4d5b1d2d63a0f21eb9c
        ViT,
        ConvNeXt, ConvMixer

# use Flux._big_show to pretty print large models
for T in (:AlexNet, :VGG, :ResNet, :GoogLeNet, :Inception3, :SqueezeNet, :DenseNet, :ResNeXt, 
<<<<<<< HEAD
          :MobileNetv2, :MobileNetv3, :MLPMixer, :ViT, :ConvNeXt)
>>>>>>> 63bcddd5514a997f8b2fdc9857b7f629e80a49fb
=======
          :MobileNetv1, :MobileNetv2, :MobileNetv3,
          :MLPMixer, :ResMLP, :gMLP, :ViT, :ConvNeXt, :ConvMixer)
>>>>>>> aba6fb832093d88dc2d2b4d5b1d2d63a0f21eb9c
>>>>>>> FluxML-master
  @eval Base.show(io::IO, ::MIME"text/plain", model::$T) = _maybe_big_show(io, model)
end

end # module
