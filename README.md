# VIVA CONTAMINATION

Automated contamination of recycle bins detection with DeepStream.

The application performs the following steps:

1. Video stream capture and decoding
2. Primary inference engine detects the contaminations in the incoming frames

The collected data can be regularly transmitted to a cloud storage.

## Usage

```
python3 detect_cont.py <config_file>
```

A sample config file is provided in `config_app_ex.txt`

## Configuration files

See the `config_app.txt` file for the different configuration options of the application. You can configure:
- multiple outputs: video file, RTSP, display;
- the data transmission: number of frames between transmissions, enabling/disabling;
- the name of the device (by defaut it is the MAC address of the first network interface;
- the source video feed.

The different models hava their own configuration file in the `config` directory:
- `config_infer_primary.txt` for the object detector;
- `config_tracker.txt` for the tracker;

If you encounter an error when trying to run the application, or if DeepStream keeps rebuiling the inference
engine files you can modify the paths in the configuration file (ie use absolute paths instead of relative ones).

## RTSP feed

If enabled, the processed video stream will be available at `localhost:8554/live`.

## Installation steps

Those steps are for a Jetson NX Xavier, but should similar for every platform.

### DeepStream 6 and its Python 3 bindings (pyds):

More information here: https://docs.nvidia.com/metropolis/deepstream/dev-guide/text/DS_Quickstart.html

### Python 3 packages

Only the `netifaces` packages is needed:
```
pip3 install netifaces paho-mqtt
```

### GstRtspserver:

```
sudo apt update
sudo apt-get install libgstrtspserver-1.0-0 gstreamer1.0-rtsp
sudo apt-get install libgirepository1.0-dev
sudo apt-get install gobject-introspection gir1.2-gst-rtsp-server-1.0
```


## Notes

- A sample `systemd` service file is provided in the `service` directory to run the application continuously.
- Starting/stopping a service at regular time: https://unix.stackexchange.com/questions/265704/start-stop-a-systemd-service-at-specific-times
- Enabling automatic security updates: https://libre-software.net/ubuntu-automatic-updates/
- If the device needs to run 24/7, it can be a good idea to schedule a daily reboot via a `cron` job.
- Compile to bytecode: `python3 -OO -m py_compile <python script>.py`
- For saving images to external drive -- check name using `lsblk`
- install nano using `sudo apt-get install nano`
- `sudo nano /etc/fstab` for the auto mount of usb on the startup

## systemd commands
- sudo systemctl enable /path/to/servicefile
- sudo systemctl start nameofservicefile.service
- sudo systemctl status nameofservicefile.service
- sudo systemctl stop nameofservicefile.service
- sudo systemctl disable nameofservicefile.service
- echo $DISPLAY to set the correct display environment in the file.

## AutoMount USB using `fstab`
- first check the USB device label and UUID by command `lsblk -f`
- Use `sudo nano /etc/fstab` command to open fstab
- Enter the information in the following format in a new line and `ctl x` for saving. \n
  `UUID=your_usb_uuid /mount/point filesystem_type options 0 0` \n
  `UUID=[Device UUID] /media/username/USB vfat defaults 0 0`
- Check if it mounted correctly using `sudo mount -a`

## Contact

Umair Iqbal - umair@uow.edu.au
Johan Barthelemy - johan@uow.edu.au
