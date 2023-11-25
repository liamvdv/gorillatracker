# import torchvision.transforms as transforms

# from gorillatracker.data_modules.data_modules import QuadletDataModule, TripletDataModule
# from gorillatracker.datasets.bristol import BristolDataset
# from gorillatracker.data_modules.transform_utils import SquarePad

# def get_bristol_transforms():
#     return transforms.Compose(
#         [
#             SquarePad(),
#             transforms.Resize(124),
#             transforms.ToTensor(),
#         ]
#     )

# class BristolTripletDataModule(TripletDataModule):
#     def get_dataset_class(self):
#         return BristolDataset

#     def get_transforms(self):
#         return get_bristol_transforms()


# class BristolQuadletDataModule(QuadletDataModule):
#     def get_dataset_class(self):
#         return BristolDataset

#     def get_transforms(self):
#         return get_bristol_transforms()
