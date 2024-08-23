import os
import numpy as np
import onnxruntime as ort

from PIL import Image

TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"
MODEL_PATH = "./model.onnx"


def load_image(path):
	image = Image.open(path)
	image = image.convert("RGB")
	image = image.resize((224, 224), Image.BILINEAR)

	image = np.array(image, dtype=np.float32)
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

	for i, image_name in enumerate(all_image_names):
		image = None
		output = None
		pred = None

		try:
			image_path = os.path.join(TEST_IMAGES_DIR, image_name)
			image = load_image(image_path)

			output = session.run([output_name], {input_name: image})
			output = output[0]

			pred = np.argmax(output, axis=1)
			pred = pred.item()

			all_preds.append(pred)
		except Exception as e:
			image_shape = None
			image_min = None
			image_max = None

			try:
				image_shape = image.shape
				image_min = image.min()
				image_max = image.max()
			except:
				pass

			raise RuntimeError(
				f"\n[i] {i}"
				f"\n[image name] {image_name}"
				f"\n[image shape] {image_shape}"
				f"\n[image min; max] {image_min}; {image_max}"
				f"\n[output] {output}"
				f"\n[error] {e}"
			)

	with open(SUBMISSION_PATH, "w") as f:
		f.write("image_name\tlabel_id\n")
		for name, cl_id in zip(all_image_names, all_preds):
			f.write(f"{name}\t{cl_id}\n")
