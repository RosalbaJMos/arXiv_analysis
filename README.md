# arXiv_Publications_Analysis
## This is an App that analyzes the abstracts of the last articles published on arXiv in 2021 and helps you to classify your own abstract into either Physics, Biology, or Computer Science

## Project Description
 Part 1:

    1.0 The First plot, shows the number of scientific articles published per month on arXiv in 2021 belonging to the main fields of Physics, Maths, Computer Science, and Biology.

 Part 2: 
 
    2.0 The App shows the keywords that appear the most when dividing the relevant field (Physics, Biology, Math, or CS) into six main topics. The main words related to each topic are shown in each plot and also displayed as wordclouds

 Part 3:
 
    3.0 In this part, the user can introduce an abstract for classification as a abstract that falls into one of the following categories: - Physics - Biology - Computer Science (CS).


## Table of Contents

1.  BIO_last_200Pubs_2021.csv --> This file contains the information (tiles, abstracts, date, and authors) of the last 200 scientific articles published on arXiv in 2021, related to the entire field of Biology. 

2. CS_last_200Pubs_2021.csv  --> This file contains the information (tiles, abstracts, date, and authors) of the last 200 scientific articles published on arXiv in 2021, related to the entire field of Computer Sciece.

3. Math_last_200Pubs_2021.csv --> This file contains the information (tiles, abstracts, date, and authors) of the last 200 scientific articles published on arXiv in 2021, related to the entire field of Math.

4. Phys_last_200Pubs_2021.csv --> This file contains the information (tiles, abstracts, date, and authors) of the last 200 scientific articles published on arXiv in 2021, related to the entire field of Physics. 

5. Pubs_PhyMatCSBIO2021.csv --> This file contains the information (Number of publications per month for each field) of the all scientific articles of Physics, Math, CS, and Biology published on arXiv in 2021.  This data is used to build the Plot 1.0.   

6. Sample_PhBiCS.csv -->  This file contains the information (Abstract, Field ) of the last 600 scientific articles published on arXiv in 2021 related to the fields of Physics, Biology, and Computer Science. This data is used to train the ML model.

7. streamlit_app_arxiv_18FEB.py  --> The python script to run on streamlit.

8. README.md

## How to Install and Run the Project
To run the app, it is necessary to have all the above-liested files inside the same directory where the streamlit_app_arxiv_18FEB.py script is located. To run it,  use the following command from your terminal:

streamlit run streamlit_app_arxiv_18FEB.py


