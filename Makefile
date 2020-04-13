clean:
	rm output_images/**/*.jpg

build:
	python camera_calibration.py

test:
	pytest