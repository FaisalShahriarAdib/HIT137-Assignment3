# HIT137-Assignment3  


Created by:

- Faisal Shahriar s390914
- Sameer Thapa s397773
- Kanij Fatema s394326
- Riad Sarkar Santo s394943

## Features Added

- Loads JPG, PNG, and BMP images.
- Displays original and modified images side by side.
- Generates 5 random non-overlapping differences.
- Uses OpenCV image processing for visual changes.
- Tracks remaining differences and player score.
- Allows maximum 3 mistakes per image.
- Shows red circles for found differences.
- Shows blue circles when Reveal All is clicked.
- Resets the game when a new image is loaded.

## Technologies Used

- Python
- Tkinter
- OpenCV
- NumPy
- Pillow

## How to Run

1. Install required libraries:

pip install opencv-python pillow numpy

2. Run the program:

python spot_the_difference.py

3. Click "Choose Image" and select an image file.

## Game Rules

- The player must find 5 hidden differences.
- Click on the modified image to identify differences.
- Correct selections are highlighted with red circles.
- Incorrect clicks increase the mistake counter.
- The player loses after 3 mistakes.
- The Reveal All button displays all remaining hidden differences.

## Object-Oriented Programming Features

This project uses object-oriented programming concepts including:

- Classes and objects
- Encapsulation
- Class methods
- Inheritance using Tkinter
- Separation of game logic and GUI components

## Image Processing Features

OpenCV is used to:

- Resize and scale images
- Apply blur effects
- Apply colour shifting
- Apply brightness adjustments
- Add random noise
- Create darkened image regions

## Future Improvements

- Add timer functionality
- Add difficulty levels
- Add multiplayer support
- Add sound effects
- Add leaderboard system
