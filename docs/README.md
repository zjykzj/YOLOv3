
# YOLOv3

## All Results

* Train 120 rounds with [yolov3_default.cfg](./config/yolov3_default.cfg) and verify with COCO val2017. Compare with other results (training 300 rounds) as follows:

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-7btt"><span style="font-style:normal">Original (darknet)</span></th>
    <th class="tg-7btt">DeNA/PyTorch_YOLOv3</th>
    <th class="tg-7btt">zjykzj/YOLOv3(This)</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">COCO AP[IoU=0.50:0.95], inference</td>
    <td class="tg-c3ow">0.310</td>
    <td class="tg-c3ow">0.311</td>
    <td class="tg-c3ow">0.315</td>
  </tr>
  <tr>
    <td class="tg-7btt">COCO AP[IoU=0.50], inference</td>
    <td class="tg-c3ow">0.553</td>
    <td class="tg-c3ow">0.558</td>
    <td class="tg-c3ow">0.543</td>
  </tr>
  <tr>
    <td class="tg-7btt">conf_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.005</td>
    <td class="tg-c3ow">0.005</td>
  </tr>
  <tr>
    <td class="tg-7btt">nms_thre</td>
    <td class="tg-c3ow">/</td>
    <td class="tg-c3ow">0.45</td>
    <td class="tg-c3ow">0.45</td>
  </tr>
  <tr>
    <td class="tg-7btt">input_size</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">416</td>
    <td class="tg-c3ow">416</td>
  </tr>
</tbody>
</table>

## Train using yolov5

[ultralytics/yolov5](https://github.com/ultralytics/yolov5) provides a very nice training framework and training recipes for different configurations of YOLOv3. The results of training are as follows

<!-- <style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-baqh{text-align:center;vertical-align:top}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
.tg .tg-7btt{border-color:inherit;font-weight:bold;text-align:center;vertical-align:top}
.tg .tg-amwm{font-weight:bold;text-align:center;vertical-align:top}
</style> -->
<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-7btt"><span style="font-style:normal">ultralytics/yolov3</span></th>
    <th class="tg-7btt"><span style="font-style:normal">ultralytics/yolov3-tiny</span></th>
    <th class="tg-amwm"><span style="font-style:normal">ultralytics/yolov3-spp</span></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-7btt">COCO AP[IoU=0.50:0.95], inference</td>
    <td class="tg-c3ow">0.450</td>
    <td class="tg-7btt">0.186</td>
    <td class="tg-amwm">0.463</td>
  </tr>
  <tr>
    <td class="tg-7btt">COCO AP[IoU=0.50], inference</td>
    <td class="tg-c3ow">0.644</td>
    <td class="tg-7btt">0.354</td>
    <td class="tg-amwm">0.657</td>
  </tr>
  <tr>
    <td class="tg-7btt">conf_thre</td>
    <td class="tg-c3ow">0.001</td>
    <td class="tg-7btt">0.001</td>
    <td class="tg-amwm">0.001</td>
  </tr>
  <tr>
    <td class="tg-7btt">nms_thre</td>
    <td class="tg-c3ow">0.6</td>
    <td class="tg-7btt">0.6</td>
    <td class="tg-amwm">0.6</td>
  </tr>
  <tr>
    <td class="tg-7btt">input_size</td>
    <td class="tg-c3ow">640</td>
    <td class="tg-c3ow">640</td>
    <td class="tg-baqh">640</td>
  </tr>
</tbody>
</table>

View [train.md](./train.md) for more detailed information.