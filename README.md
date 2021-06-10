# Disaster Response Pipeline Project

[homepage](screenshots/homepage.PNG)

##Description
### Instructions:
1. Clone the repository by executing <code>git clone https://github.com/siddarthaThentu/Disaster-Response-Pipeline.git<code>

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Project Post Mortem:

This project consists of two parts.
1. Front-end -> Developed with bootstrap, html and css
2. Back-end -> A flask web server which gets executed when _python run.py_ is run

