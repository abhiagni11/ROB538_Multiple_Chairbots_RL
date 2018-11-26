import os
try:
    original_umask = os.umask(0)
    os.makedirs('models/vdn', 0777)
finally:
    os.umask(original_umask)