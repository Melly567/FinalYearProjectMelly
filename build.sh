<<<<<<< HEAD
#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
=======
#!/usr/bin/env bash
# exit on error
set -o errexit

pip install -r requirements.txt

python manage.py collectstatic --no-input
>>>>>>> 2d5f4ea00d3443d5b05515e1c97230920c79bf22
python manage.py migrate