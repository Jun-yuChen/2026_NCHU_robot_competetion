# optoforce_driver

OptoForce HEX-70-CE-2000N 六軸力/扭矩感測器的**非官方**ROS2(Humble, ament_python)驅動節點。

原廠沒有提供ROS/ROS2套件，這個節點直接照官方協定文件(`OptoForce General DAQ -
protocol description 1.4.1`，USB/CAN/UART通用)自己解析USB-CDC虛擬序列埠傳來的
22-byte資料封包，換算成N/Nm後發布成標準的`geometry_msgs/WrenchStamped`。

## ⚠️ 校正係數是綁定特定感測器序號的

`optoforce_node.py`裡的`SENSITIVITY`那組counts轉N/Nm係數，是**序號ICE042**這一顆
感測器出廠校正報告(`SensitivityReport`)裡的數字。如果你用的是**同一顆實體感測器**
(只是換一台電腦跑)，可以直接用；如果是**另一顆感測器**(哪怕同型號)，一定要換成
你那顆的校正報告裡的數字，否則讀出來的N/Nm數值是錯的。

## 安裝

需要 ROS2 Humble 環境(有`rclpy`、`geometry_msgs`)，以及：

```bash
pip install pyserial
# 或
sudo apt install python3-serial
```

把這個資料夾clone進你的ROS2 workspace的`src/`底下，然後build：

```bash
cd ~/your_ws/src
git clone https://github.com/keep5lience5555-crypto/optoforce.git optoforce_driver
cd ~/your_ws
colcon build --packages-select optoforce_driver
source install/setup.bash
```

## 接上感測器

### 原生 Linux（Ubuntu 直接裝機，不是WSL）

USB接上後直接會出現`/dev/ttyACM0`(或類似編號，看你機器上還有沒有其他USB序列裝置)：

```bash
ls /dev/ttyACM*
```

### WSL

WSL看不到USB裝置，需要先用[usbipd-win](https://github.com/dorssel/usbipd-win)把裝置
從Windows轉接過去。**在Windows PowerShell(系統管理員)**執行：

```powershell
usbipd list                          # 找到OptoForce的BUSID(通常是VID 04d8:000a)
usbipd bind --busid <你的busid>      # 第一次要bind，之後不用重複
usbipd attach --wsl --busid <你的busid>
```

每次重新插拔USB或重開WSL，`attach`這步都要重做一次(`bind`只要做一次)。回WSL確認：

```bash
ls /dev/ttyACM*
```

### 序列埠權限（Linux共通，跟WSL/原生無關）

第一次會遇到`Permission denied`，兩種解法擇一：

```bash
# 臨時：每次重新插拔/attach裝置都要重做一次
sudo chmod 666 /dev/ttyACM0

# 一勞永逸：把帳號加進dialout群組(要重新登入或重開終端機才生效)
sudo usermod -aG dialout $USER
```

## 執行

```bash
ros2 run optoforce_driver optoforce_node --ros-args -p port:=/dev/ttyACM0
```

`port`參數預設就是`/dev/ttyACM0`，如果你的裝置編號不同要自己指定。

另開一個終端機確認有沒有收到資料：

```bash
ros2 topic echo /optoforce/wrench
```

夾爪/感測器沒有受力時，`force`/`torque`應該接近0。

## 歸零(Zero)

感測器的歸零設定**斷電就會重置**，這個node開機時會自動送一次硬體歸零指令
(見下方協定細節)。如果跑一段時間後想重新歸零(不用重開整個node)：

```bash
ros2 service call /optoforce/zero std_srvs/srv/Trigger {}
```

⚠️ 呼叫這個service的當下，感測器不能受力，不然新的零點基準會歪掉。

沒有做成獨立的另一支歸零腳本，是因為序列埠同一時間只能被一個process開著——
這個service是由已經握有序列埠的node自己執行送出，呼叫端不需要、也不能自己
直接連序列埠。

## 發布的topic

- `/optoforce/wrench` (`geometry_msgs/WrenchStamped`)：`wrench.force.{x,y,z}`單位N，
  `wrench.torque.{x,y,z}`單位Nm。

## 協定細節（想自己核對/換感測器要改的地方）

- 22-byte封包：`Header(170,7,8,16)` + `SampleCounter(UINT16)` + `Status(UINT16)`
  + `Fx,Fy,Fz,Tx,Ty,Tz`(各`INT16`) + `Checksum(UINT16)`，全部big-endian。
- `Checksum` = 前面20個bytes的總和(取UINT16)。
- `力/扭矩[N或Nm] = counts / 校正報告裡的(Counts@N.C.) * (N.C.)`，等同直接除以
  校正報告裡給的`Counts/N`(或`Counts/Nm`)欄位，即`optoforce_node.py`裡的
  `SENSITIVITY`字典。
- 感測器開機預設100Hz持續傳送資料，可用9-byte CONFIGURATION封包調整更新率/濾波
  截止頻率/歸零，這個節點目前沒有實作送CONFIGURATION封包(用預設值)。

詳細封包格式、CONFIGURATION/STATUS欄位定義，請參考原廠協定PDF(隨機器出貨附的
隨身碟內容，或聯絡OptoForce/OnRobot取得)。
