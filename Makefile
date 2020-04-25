clean:
	rm output_images/**/*.jpg
	rm output_videos/*.mp4

build:
	python camera_calibration.py
	python perspective.py
	python binary_image.py
	python pipeline.py

test:
	pytest