import glob
import os
#from zipfile import ZipFile
from datetime import date
import subprocess
from subprocess import check_output, CalledProcessError, STDOUT

class DownloadExractor():
    def __init__(self, today_folder):
        self.today_folder = today_folder
        self.find_latest_zip_file()

    def build_today_folder(self):
        if not os.path.exists(self.today_folder):
            os.makedirs(self.today_folder)

    def unzip_file(self):
        #with ZipFile(self.latest_zip_file, 'r') as f:
        #    f.extractall(self.today_folder)
        check_output(['unzip', self.latest_zip_file, '-d', self.today_folder], stderr=STDOUT)

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
        self.base_test_py = os.path.join(self.config_path, "tests", "its_base_test.py")

        self.devices = None
        self.tablet_name = None

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

    def fix_filepath_error(self):
        replace_1 = "file://mnt"
        replace_2 = "sdcard"

        with open(self.utils_session_py) as file:
            data = file.read()

        data = data.replace(replace_1, "", 1)
        data = data.replace(replace_2, "storage/emulated/0")

        with open(self.utils_session_py, "w") as file:
            file.writelines(data)

    def fix_rotation_error(self):
        replace_1 = "elif 'ROTATION_0' in landscape_val:"
        target_1 = "elif 'ROTATION_0' in landscape_val: landscape_val = '0'"
        replace_2 = "  landscape_val = '0'"
        target_2 = "elif 'ROTATION_270' in landscape_val: landscape_val = '3'"

        with open(self.base_test_py) as file:
            data = file.read()

        data = data.replace(replace_1, target_1, 1)
        data = data.replace(replace_2, target_2, 1)

        with open(self.base_test_py, "w") as file:
            file.writelines(data)

    def find_devices(self):
        devices_list_unencoded = subprocess.check_output("adb devices", stderr=subprocess.STDOUT, shell=True)
        devices_list = str(devices_list_unencoded.decode('utf-8')).strip()
        devices = devices_list.split("\n")[1:]
        if len(devices) != 2:
            raise ValueError(f'device number is not correct.')
        self.devices = [device.split("\t")[0] for device in devices]

        get_device_name = "adb -s " + self.devices[1] + " shell getprop ro.product.device"
        tablet_name_unencoded = subprocess.check_output(get_device_name, stderr=subprocess.STDOUT, shell=True)
        self.tablet_name = str(tablet_name_unencoded.decode('utf-8')).strip()

    
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
    print("Have built folder: " + DE.today_folder + "\n")
    DE.unzip_file()
    print("Have extracted file: " + DE.latest_zip_file + "\n")

    # finding and checking target device and tablet SN number
    SI.find_devices()
    print("target device: " + SI.devices[0] + ", tablet: " + SI.devices[1] + ", tablet name: " + SI.tablet_name + "\n")

    # editing config
    SI.edit_config()
    print("Have edited " + SI.config_file + "\n")

    # fixing error
    if SI.tablet_name == "tangorpro":
        SI.fix_tablet_version_error()
        SI.fix_filepath_error()
        print("Have fixed " + SI.utils_session_py)

        SI.fix_rotation_error()
        print("Have fixed " + SI.base_test_py + "\n")

    # print commands for running all its
    CP.print_command(SI.devices)
