import subprocess
import os

def get_version():
    major = ''
    minor = ''
    pip_output = subprocess.run(["pip","list"],capture_output=True,encoding='utf-8')
    lines = pip_output.stdout.split("\n")
    for line in lines:
        if line.startswith("scikit-learn-intelex"):
            version = line.split()[1].split(".")
            major = version[0]
            minor = version[1]

    return major + "." + minor
