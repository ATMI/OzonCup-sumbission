import torch
from torchvision import transforms
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from baseline import init_model, BaseDataset

MODEL_WEIGHTS = "baseline.pth"
TEST_DATASET = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = init_model(device)
    model.load_state_dict(torch.load(MODEL_WEIGHTS))
    model.eval()

    img_size = 224
    trans = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dset = BaseDataset(TEST_DATASET, transform=trans)
    batch_size = 16
    num_workers = 4
    testloader = DataLoader(dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_image_names = [os.path.basename(path) for path in dset.image_paths]
    all_preds = []
    model = model.eval()
    with torch.no_grad():
        for images, _ in testloader:
            images = images.to(device)
            outputs = model(images).squeeze()
            preds = torch.sigmoid(outputs) >= 0.5
            all_preds.extend(preds.cpu().numpy().astype(int).tolist())

    with open(SUBMISSION_PATH, "w") as f:
        f.write("image_name\tlabel_id\n")
        for name, cl_id in zip(all_image_names, all_preds):
            f.write(f"{name}\t{cl_id}\n")
