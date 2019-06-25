Requisiti per poter eseguire il programma: 
	0a) installare opencv con brew  #installa la libreria opencv
		$ brew install opencv
	0b) installare pkg-config con brew  #serve per trovare la libreria opencv
		$ brew install pkg-config
	0c) installare openMP con brew  #installa la libreria omp
		$ brew install libomp
------------------------------------------------
Eseguire da terminale:
	-Sequenziale
		1) Cambiare nel main.cpp il percorso dell'immagine. Sostituire all'interno della funzione main:
			Mat image = imread("../img/batman.jpg", IMREAD_COLOR);  con  Mat image = imread("./img/batman.jpg", IMREAD_COLOR);
		2) $ g++ -o prova main.cpp `pkg-config --cflags --libs opencv4` -std=c++11
		3) Una volta generato l'eseguibile di nome prova digitare:
			$ ./prova

	-OpenMP
		1) Cambiare nel main.cpp il percorso dell'immagine. Sostituire all'interno della funzione main:
			Mat image = imread("../img/batman.jpg", IMREAD_COLOR);  con  Mat image = imread("./img/batman.jpg", IMREAD_COLOR);
		2) $ clang++ -Xpreprocessor -fopenmp main.cpp -o prova -lomp `pkg-config --cflags --libs opencv4` -std=c++11
		3) Una volta generato l'eseguibile di nome prova digitare:
			$ ./prova
------------------------------------------------
Eseguire con CLion:
	-Sequenziale:  Aprire il progetto Sequential 
	-OpenMP: Aprire il progetto OpenMP
------------------------------------------------
Settare Cuda su linux (computer micc)
	1) creare progetto cuda 
	2) settare il CUDA toolkit con la versione 10.1
		project<< C/C++ Build << CUDA Toolkit << selezionare la versione 10.1
	3) Settare l' NVCC linker e compiler
		project<< C/C++ Build << Settings
			a) In NVCC linker << Libreries ,nel tab Libraries (-l)aggiungere:
				opencv_highgui
				opencv_imgcodecs
				opencv_core 
				opencv_imgproc
			mentre nel tab Library search path (-L) aggiungere:
				/usr/lib

			b) In NVCC Compiler ,nel tab Include path (-I) aggiungere:
				/usr/local/include
				/usr/local/include/opencv-3.4.3
------------------------------------------------
Per eseguire il codice aprire il progetto histogram_equalization_CUDA con Eclipse