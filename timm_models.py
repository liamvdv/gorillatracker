import timm

models = timm.list_models("*resnet*", pretrained=True)
for model in models:
    print(model)
