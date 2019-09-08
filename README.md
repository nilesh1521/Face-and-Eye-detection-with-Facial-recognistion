# Face-and-Eye-detection-with-Facial-recognistion

• OpenCV uses machine learning algorithms to search for faces within a picture. Because faces are so complicated, 
there isn’t one simple test that will tell you if it found a face or not. Instead, there are thousands of small patterns and 
features that must be matched. The algorithms break the task of identifying the face into thousands of smaller, bite-sized tasks, each 
of which is easy to solve.

• For face detection, the algorithm starts at the top left of a picture and moves down across small blocks of data, looking at each 
block, constantly asking, “Is this a face? … Is this a face? … Is this a face?”

• To get around this, OpenCV uses cascades. the OpenCV cascade breaks the problem of detecting faces into multiple stages. For each 
block, it does a very rough and quick test. If that passes, it does a slightly more detailed test, and so on. The algorithm may have 
30 to 50 of these stages or cascades, and it will only detect a face if all stages pass. The advantage is that most of the picture 
will return a negative during the first few stages, which means the algorithm won’t waste time testing all 6,000 features on it. 
Instead of taking hours, face detection can now be done in real-time
