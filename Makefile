update-readme-assets:
	cp artifacts/shap_summary.png images/shap_summary.png
	cp artifacts/shap_beeswarm.png images/shap_beeswarm.png

update-deps:
	pip-compile requirements.in --output-file requirements.txt