- The folders in this directory each represent the map considered except the folders 'Extra' and 'Scale 1 Paths'
- The naming convention of the folder si as follows:
	- Folder name 'X_Y__obs_0.p' is contains the results of the map with:
		- Size X x Y nodes/grids
		- Obstacle Density of p percent
- In each Folder there is csv file with the same name as the folder and same naming convention.
- The csv file is the record of the 50 trials of RRT algorithm on the respective map
- The png images have the name of the folder as above with _pathx added on to the end where x ranges from 0 to 49 and represent the trial number which generated that map.

To run the python script:
pip install pygame
pip install heapdict
pip install pandas
pip install numpy
pip install networkx


python RRT_Robotics_Project_Final_Code.py