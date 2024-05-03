# Copyright 2024 The Android Open Source Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Utility functions for interacting with a device via the UI."""

import datetime
import logging
import time
import types

import camera_properties_utils
import its_device_utils

_PERMISSIONS_LIST = ('CAMERA', 'RECORD_AUDIO', 'ACCESS_FINE_LOCATION',
                     'ACCESS_COARSE_LOCATION', 'WRITE_EXTERNAL_STORAGE',
                     'READ_EXTERNAL_STORAGE')

ACTION_ITS_DO_JCA_CAPTURE = (
    'com.android.cts.verifier.camera.its.ACTION_ITS_DO_JCA_CAPTURE'
)
ACTIVITY_WAIT_TIME_SECONDS = 5
CAPTURE_BUTTON_RESOURCE_ID = 'CaptureButton'
DONE_BUTTON_TXT = 'Done'
FLASH_MODE_TO_CLICKS = types.MappingProxyType({
    'OFF': 3,
    'AUTO': 2
})
IMG_CAPTURE_CMD = 'am start -a android.media.action.STILL_IMAGE_CAPTURE'
ITS_ACTIVITY_TEXT = 'Camera ITS Test'
JPG_FORMAT_STR = '.jpg'
OK_BUTTON_TXT = 'OK'
TAKE_PHOTO_CMD = 'input keyevent KEYCODE_CAMERA'
QUICK_SETTINGS_RESOURCE_ID = 'QuickSettingDropDown'
QUICK_SET_FLASH_RESOURCE_ID = 'QuickSetFlash'
QUICK_SET_FLIP_CAMERA_RESOURCE_ID = 'QuickSetFlipCamera'
REMOVE_CAMERA_FILES_CMD = 'rm sdcard/DCIM/Camera/*'
UI_DESCRIPTION_BACK_CAMERA = 'Back Camera'
UI_DESCRIPTION_FRONT_CAMERA = 'Front Camera'
UI_OBJECT_WAIT_TIME_SECONDS = datetime.timedelta(seconds=3)
VIEWFINDER_NOT_VISIBLE_PREFIX = 'viewfinder_not_visible'
VIEWFINDER_VISIBLE_PREFIX = 'viewfinder_visible'
WAIT_INTERVAL_FIVE_SECONDS = datetime.timedelta(seconds=5)


def _find_ui_object_else_click(object_to_await, object_to_click):
  """Waits for a UI object to be visible. If not, clicks another UI object.

  Args:
    object_to_await: A snippet-uiautomator selector object to be awaited.
    object_to_click: A snippet-uiautomator selector object to be clicked.
  """
  if not object_to_await.wait.exists(UI_OBJECT_WAIT_TIME_SECONDS):
    object_to_click.click()


def verify_ui_object_visible(ui_object, call_on_fail=None):
  """Verifies that a UI object is visible.

  Args:
    ui_object: A snippet-uiautomator selector object.
    call_on_fail: [Optional] Callable; method to call on failure.
  """
  ui_object_visible = ui_object.wait.exists(UI_OBJECT_WAIT_TIME_SECONDS)
  if not ui_object_visible:
    if call_on_fail is not None:
      call_on_fail()
    raise AssertionError('UI object was not visible!')


def open_jca_viewfinder(dut, log_path):
  """Sends an intent to JCA and open its viewfinder.

  Args:
    dut: An Android controller device object.
    log_path: str; log path to save screenshots.
  Raises:
    AssertionError: If JCA viewfinder is not visible.
  """
  its_device_utils.start_its_test_activity(dut.serial)
  call_on_fail = lambda: dut.take_screenshot(log_path, prefix='its_not_found')
  verify_ui_object_visible(
      dut.ui(text=ITS_ACTIVITY_TEXT),
      call_on_fail=call_on_fail
  )

  # Send intent to ItsTestActivity, which will start the correct JCA activity.
  its_device_utils.run(
      f'adb -s {dut.serial} shell am broadcast -a {ACTION_ITS_DO_JCA_CAPTURE}'
  )
  jca_capture_button_visible = dut.ui(
      res=CAPTURE_BUTTON_RESOURCE_ID).wait.exists(
          UI_OBJECT_WAIT_TIME_SECONDS)
  if not jca_capture_button_visible:
    dut.take_screenshot(log_path, prefix=VIEWFINDER_NOT_VISIBLE_PREFIX)
    logging.debug('Current UI dump: %s', dut.ui.dump())
    raise AssertionError('JCA was not started successfully!')
  dut.take_screenshot(log_path, prefix=VIEWFINDER_VISIBLE_PREFIX)


def switch_jca_camera(dut, log_path, facing):
  """Interacts with JCA UI to switch camera if necessary.

  Args:
    dut: An Android controller device object.
    log_path: str; log path to save screenshots.
    facing: str; constant describing the direction the camera lens faces.
  Raises:
    AssertionError: If JCA does not report that camera has been switched.
  """
  if facing == camera_properties_utils.LENS_FACING['BACK']:
    ui_facing_description = UI_DESCRIPTION_BACK_CAMERA
  elif facing == camera_properties_utils.LENS_FACING['FRONT']:
    ui_facing_description = UI_DESCRIPTION_FRONT_CAMERA
  else:
    raise ValueError(f'Unknown facing: {facing}')
  dut.ui(res=QUICK_SETTINGS_RESOURCE_ID).click()
  _find_ui_object_else_click(dut.ui(desc=ui_facing_description),
                             dut.ui(res=QUICK_SET_FLIP_CAMERA_RESOURCE_ID))
  if not dut.ui(desc=ui_facing_description).wait.exists(
      UI_OBJECT_WAIT_TIME_SECONDS):
    dut.take_screenshot(log_path, prefix='failed_to_switch_camera')
    logging.debug('JCA UI dump: %s', dut.ui.dump())
    raise AssertionError(f'Failed to switch to {ui_facing_description}!')
  dut.take_screenshot(
      log_path, prefix=f"switched_to_{ui_facing_description.replace(' ', '_')}"
  )
  dut.ui(res=QUICK_SETTINGS_RESOURCE_ID).click()


def default_camera_app_setup(device_id, pkg_name):
  """Setup Camera app by providing required permissions.

  Args:
    device_id: serial id of device.
    pkg_name: pkg name of the app to setup.
  Returns:
    Runtime exception from called function or None.
  """
  logging.debug('Setting up the app with permission.')
  for permission in _PERMISSIONS_LIST:
    cmd = f'pm grant {pkg_name} android.permission.{permission}'
    its_device_utils.run_adb_shell_command(device_id, cmd)
  allow_manage_storage_cmd = (
      f'appops set {pkg_name} MANAGE_EXTERNAL_STORAGE allow'
  )
  its_device_utils.run_adb_shell_command(device_id, allow_manage_storage_cmd)


def pull_img_files(device_id, input_path, output_path):
  """Pulls files from the input_path on the device to output_path.

  Args:
    device_id: serial id of device.
    input_path: File location on device.
    output_path: Location to save the file on the host.
  """
  logging.debug('Pulling files from the device')
  pull_cmd = f'adb -s {device_id} pull {input_path} {output_path}'
  its_device_utils.run(pull_cmd)


def launch_and_take_capture(dut, pkg_name):
  """Launches the camera app and takes still capture.

  Args:
    dut: An Android controller device object.
    pkg_name: pkg_name of the default camera app to
      be used for captures.

  Returns:
    img_path_on_dut: Path of the captured image on the device
  """
  device_id = dut.serial
  try:
    logging.debug('Launching app: %s', pkg_name)
    launch_cmd = f'monkey -p {pkg_name} 1'
    its_device_utils.run_adb_shell_command(device_id, launch_cmd)

    # Click OK/Done button on initial pop up windows
    if dut.ui(text=OK_BUTTON_TXT).wait.exists(
        timeout=WAIT_INTERVAL_FIVE_SECONDS):
      dut.ui(text=OK_BUTTON_TXT).click.wait()
    if dut.ui(text=DONE_BUTTON_TXT).wait.exists(
        timeout=WAIT_INTERVAL_FIVE_SECONDS):
      dut.ui(text=DONE_BUTTON_TXT).click.wait()

    its_device_utils.run_adb_shell_command(device_id, IMG_CAPTURE_CMD)
    time.sleep(ACTIVITY_WAIT_TIME_SECONDS)
    logging.debug('Taking photo')
    its_device_utils.run_adb_shell_command(device_id, TAKE_PHOTO_CMD)
    time.sleep(ACTIVITY_WAIT_TIME_SECONDS)
    img_path_on_dut = dut.adb.shell(
        'find {} ! -empty -a ! -name \'.pending*\' -a -type f'.format(
            '/sdcard/DCIM/Camera')).decode('utf-8').strip()
    logging.debug('Image path on DUT: %s', img_path_on_dut)
    if JPG_FORMAT_STR not in img_path_on_dut:
      raise AssertionError('Failed to find jpg files!')
  finally:
    force_stop_app(dut, pkg_name)
  return img_path_on_dut


def force_stop_app(dut, pkg_name):
  """Force stops an app with given pkg_name.

  Args:
    dut: An Android controller device object.
    pkg_name: pkg_name of the app to be stopped.
  """
  logging.debug('Closing app: %s', pkg_name)
  force_stop_cmd = f'am force-stop {pkg_name}'
  dut.adb.shell(force_stop_cmd)


def default_camera_app_dut_setup(device_id, pkg_name):
  """Setup the device for testing default camera app.

  Args:
    device_id: serial id of device.
    pkg_name: pkg_name of the app.
  Returns:
    Runtime exception from called function or None.
  """
  default_camera_app_setup(device_id, pkg_name)
  its_device_utils.run_adb_shell_command(device_id, REMOVE_CAMERA_FILES_CMD)
