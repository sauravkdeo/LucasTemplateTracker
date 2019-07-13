# **Lucas Kanade Tracker**

---

The goals / steps of this project is to implement Lucas Kanade template tracker algorithm

---
### 1. Results

<p align="center">
<img src="https://github.com/sauravkdeo/LucasTemplateTracker/blob/master/output/car.gif" width="400" height="270" border="10">
</p>
<p align="center">
<img src="https://github.com/sauravkdeo/LucasTemplateTracker/blob/master/output/Human.gif" width="400" height="270" border="10">
</p>
<p align="center">
<img src="https://github.com/sauravkdeo/LucasTemplateTracker/blob/master/output/Vase.gif" width="400" height="270" border="10">
</p>

### 2. Pipeline

My algorithm can be easily explained using the undementioned image.


<p align="center">
<img src="https://github.com/sauravkdeo/LucasTemplateTracker/blob/master/output/lucas_kanade.png" width="400" height="270" border="10">
</p>


### 3. Instructions to run the tracker

  - Paste the `data` folder in the same folder as the code.  

  - Run the undermentioned command to run the LK_Affine_Tracker.
	```
	python3 ./LK_Affine_Tracker
	```

  - The code takes option from the user which video to run. 

  - To select the ROI click,drag and drop the marker over the desired ROI and then press enter. 
