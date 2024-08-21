import os
import cv2
import numpy as np
import onnxruntime as ort

TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"
MODEL_PATH = "./model.ort"


def load_image(path):
	image = cv2.imread(path)
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
	image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)

	image = image.astype(np.float32)
	image = image / 255
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	image = (image - mean) / std

	image = image.transpose(2, 0, 1)
	image = np.expand_dims(image, axis=0)

	return image


if __name__ == "__main__":
	all_image_names = os.listdir(TEST_IMAGES_DIR)
	all_preds = []

	session = ort.InferenceSession(MODEL_PATH)
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name

	for image_name in all_image_names:
		image_path = os.path.join(TEST_IMAGES_DIR, image_name)
		image = load_image(image_path)

		output = session.run([output_name], {input_name: image})[0]
		output = output[0].item()
		pred = int(output > 0.5)
		all_preds.append(pred)

	with open(SUBMISSION_PATH, "w") as f:
		f.write("image_name\tlabel_id\n")
		for name, cl_id in zip(all_image_names, all_preds):
			f.write(f"{name}\t{cl_id}\n")
