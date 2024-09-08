import os
import time
import numpy as np
import onnxruntime as ort

from PIL import Image
from threading import Thread
from queue import Queue

TEST_IMAGES_DIR = "./data/test/"
SUBMISSION_PATH = "./data/submission.csv"
MODEL_PATH = "./model.onnx"


def load_image(path):
	image = Image.open(path)
	if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
		image = image.convert("RGBA")
		background = Image.new("RGB", image.size, (255, 255, 255))
		background.paste(image, mask=image.split()[3])
		image = background
	else:
		image = image.convert("RGB")
	image = image.resize((384, 384), Image.BILINEAR)

	image = np.array(image, dtype=np.float32)
	image = image / 255
	mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
	std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
	image = (image - mean) / std

	image = image.transpose(2, 0, 1)
	image = np.expand_dims(image, axis=0)

	return image


def load_images(images_dir: str, queue: Queue):
	all_image_names = os.listdir(images_dir)

	for i, image_name in enumerate(all_image_names):
		try:
			image_path = os.path.join(images_dir, image_name)
			image = load_image(image_path)
			queue.put((image_name, image))
		except Exception as e:
			# TODO: proper exception handling
			print(e)

	queue.put(None)


if __name__ == "__main__":
	image_queue = Queue(2)
	loader_thread = Thread(target=load_images, args=(TEST_IMAGES_DIR, image_queue))
	loader_thread.start()

	all_preds = []

	session = ort.InferenceSession(MODEL_PATH)
	input_name = session.get_inputs()[0].name
	output_name = session.get_outputs()[0].name

	start_time = time.time()

	while True:
		data = image_queue.get()
		if data is None:
			break

		image_name, image = data
		output = session.run([output_name], {input_name: image})
		output = output[0]

		pred = output > 0.5
		pred = pred.item()
		pred = int(pred)
		pred = (image_name, pred)

		all_preds.append(pred)

	end_time = time.time()
	exec_time = end_time - start_time
	print(exec_time)

	with open(SUBMISSION_PATH, "w") as f:
		f.write("image_name\tlabel_id\n")
		for name, cl_id in all_preds:
			f.write(f"{name}\t{cl_id}\n")

	loader_thread.join()
