update_dependencies:
	pip install pip-tools
	pip-compile
	pip install -r requirements.txt