### Project description

This project contains code for fitting 2D maps generated by SLAM algorithms to the ground truth map. More generally, it can be used to fit any map to another version of it that has been translated and/or rotated. Here we do not account for reflections because we consider a reflected map to be a different floorplan.


### Interface

There are 3 main inputs used to run the code. One is the image of the ground truth file, another is a json file with the corners of the ground truth (you can find example of such a file in data/ground_truth.json, as well as an explanation of it in the Data section of this README) and the last one is the image of the map to be fitted to the ground truth.

To run the code using the files in data/ folder use the following commands in the command line:

```
cd source
fit_map.py from_gt_img_and_corners_file ../data/ground_truth.json ../data/ground_truth.png ../data/cartographer-slam-floorplan-fast-v3.pgm --show
```

It is also possible to run the code if you only have the corners file and the target floorplan image:

```
fit_map.py from_gt_corners ../data/ground_truth.json ../data/cartographer-slam-floorplan-fast-v3.pgm --show
```
The above command will invoke constructing the ground truth image from the json file that will then be used to evaluate various transformations for the input floorplan.

If you just have two maps and no corners/polylines file you can use the following command format:

```
fit_map.py from_gt_img ../data/ground_truth.png ../data/cartographer-slam-floorplan-fast-v3.pgm --show
```
This will run corner detection on ground_truth.png and use the detected corners to run the comparison. This will not generate the complete ground truth json with all the corners. It would be possible to write code that would create the correct ground_truth.json for the ground truth file such as data/ground_truth.png that is a perfect, manually generated file. You can read more about this at link. However, one does not need to get all the corners in the map for the fitting to be possible. As long as we get two corners in one of the maps and their corresponding two corners in the other map (and that these corners are not very close to each other), the fitting fill provide a correct result. 

#### Options

There are several options that the code can be run with.

##### --show

This option displays the fitted map overlaid on the ground truth image at the end.

##### --same_scale

This option signals that both maps have the same scale. This reduces search space and time needed to run the code.

##### --crop_gt

This options crops the ground truth image so there is only a small padding around useful information in the image.

There is an implicit assumption in the code that the ground truth image is cropped. This assumption was made to reduce the search space and decrease running time. It is expressed as an if condition in fit_map.py.
```
# If after fitting the floorplan is too small, then skip evaluation, it is not the right transform.
# This reduces search space.
ratio = resized_floorplan_area/gt_img_area
if ratio <= MIN_AREA_PCTG:
		continue
```

You can use this option to crop your ground truth. If you are comparing two noisy maps and the first map has some outlier pixels that make the image larger, you might not get the image as cropped as it should be. The image would need to be denoised first before cropping.

##### --crop_target

 This option crops the floorplan so that there is not too much irrelevant space around the informative part of the image.

### Data

data/ folder contains the ground truth .json file, the image of the ground truth and several maps I collected by running 2D lidar SLAM in a part of my apartment.

#### Ground truth corners file format

Ground truth json file is a file that contains a full description of the area that is being fitted. The area is described as a set of polylines, each polyline described by a sequence of points that connect to each other consecutively. If the polyline describes a loop in the map, then the first and the last point of the polyline are the same. Here is an illustration of what a file like that could look like:

```
{"polylines": 
{"0": [[5, 5,], [5, 600], [700, 600], [700, 5], [5, 5]], 
"1": [[700, 600], [700, 1200], [200, 1200], [200, 600]] }
```

### Examples

Here are some successful examples of fitting.

[[figures/figure-good-floorplan.png]]
[[figures/figure-bad-floorplan.png]]

As we can see in both cases, the floorplan was fitted closely (as much as possible) to the ground truth. The second figure is a bit more interesting because the floorplan itself is quite bad, deformed and incomplete. However, the algorithm succeeds in fitting it to the ground truth in a reasonable fashion.
