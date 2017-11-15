# Use an official Python runtime as a parent image
FROM python:3.6.3

# Set the working directory to /app
WORKDIR /app/python

# Copy the current directory contents into the container at /app
ADD ./data /app/data
ADD ./python /app/python
ADD ./requirements.txt /app/python

# Install any needed packages specified in requirements.txt
RUN pip install -r /app/python/requirements.txt

# Run script on launch
CMD bash ./run_script.sh
