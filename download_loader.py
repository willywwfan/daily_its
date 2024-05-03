import glob
import os
from zipfile import ZipFile
from datetime import date
import subprocess
import yaml

class DownloadExractor():
    def __init__(self, today_folder):
        self.today_folder = today_folder
        self.find_latest_zip_file()

    def build_today_folder(self):
        if not os.path.exists(self.today_folder):
            os.makedirs(self.today_folder)

    def unzip_file(self):
        with ZipFile(self.latest_zip_file, 'r') as f:
            f.extractall(self.today_folder)

    def find_latest_zip_file(self):
        downloads_path = os.path.join("/home", os.getlogin(), "Downloads")
        zip_files = glob.glob(os.path.join(downloads_path, "*.zip"))
        self.latest_zip_file = max(zip_files, key = os.path.getctime)

class ScriptInitialer():
    def __init__(self, today_folder):
        self.today_folder = today_folder
        self.top_path = os.path.join(self.today_folder, "android-cts-verifier")

        self.config_path = os.path.join(self.top_path, "CameraITS")
        self.config_file = os.path.join(self.config_path, "config.yml")

        self.util_path = os.path.join(self.config_path, "utils")
        self.utils_session_py = os.path.join(self.util_path, "its_session_utils.py")

        self.devices = None

    def edit_config(self):
        with open(self.config_file) as file:
            data = file.readlines()

        for i in range(len(data)):
            if "<device-id>" in data[i]:
                data[i] = data[i].replace("<device-id>", self.devices[0])

            if "<tablet-id>" in data[i]:
                data[i] = data[i].replace("<tablet-id>", self.devices[1])
                
            if "<camera-id>" in data[i]:
                data[i] = data[i].replace("<camera-id>", "0")
                break

        with open(self.config_file, 'w') as file:
            file.writelines(data)

    def fix_tablet_version_error(self):
        error_part = "raise AssertionError(TABLET_NOT_ALLOWED_ERROR_MSG)"
        
        with open(self.utils_session_py) as file:
            data = file.read()

        data = data.replace(error_part, 'tablet_name = "gta4lwifi"', 1)

        with open(self.utils_session_py, "w") as file:
            file.writelines(data)

    def find_devices(self):
        out_raw = subprocess.check_output("adb devices", stderr=subprocess.STDOUT, shell=True)
        out_s = str(out_raw.decode('utf-8')).strip()
        devices = out_s.split("\n")[1:]
        if len(devices) != 2:
            raise ValueError(f'device number is not correct.')
        self.devices = [device.split("\t")[0] for device in devices]

    
class CommandPrinter():
    def __init__(self, today_folder):
        self.today_folder = today_folder
    def print_command(self, devices):
        cd_top = "cd " + os.path.join(self.today_folder, "android-cts-verifier") + ";"
        install_apk = "adb -s " + devices[0] + " install -r -g CtsVerifier.apk;"
        activate_conda = "conda activate its_env_vic;"
        cd_ITS = "cd CameraITS;"
        source_env = "source build/envsetup.sh;"
        run_test = "python tools/run_all_tests.py;"
        print(cd_top + install_apk + activate_conda + cd_ITS + source_env + run_test)

if __name__ == "__main__":
    today_folder = str(date.today())
    DE = DownloadExractor(today_folder)
    SI = ScriptInitialer(today_folder)
    CP = CommandPrinter(today_folder)

    # build folder and extract its file
    DE.build_today_folder()
    print("Have built folder: " + DE.today_folder)
    DE.unzip_file()
    print("Have extracted file: " + DE.latest_zip_file + "\n")

    SI.find_devices()
    print("target device: " + SI.devices[0] + ", tablet: " + SI.devices[1])
    SI.edit_config()
    print("Have edited " + SI.config_file)
    SI.fix_tablet_version_error()
    print("Have fixed " + SI.utils_session_py + "\n")

    CP.print_command(SI.devices)