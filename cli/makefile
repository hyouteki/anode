all: anodecli

anodecli_OF-CNNLSTM:
	$(TIME) python anodecli_OF-CNNLSTM.py Explosion ./videos/arson011_x264.mp4 ./out/arson011_OF_CNNLSTM.pkl

anodecli_CNN:
	$(TIME) python anodecli_SNN.py arson OF-CNN ./videos/arson011_x264.mp4 ./out/arson011_OF-CNN.pkl

anodecli_SNN:
	$(TIME) python anodecli_SNN.py arson OF-SNN ./videos/arson011_x264.mp4 ./out/arson011_OF-CNN.pkl

anodecli_SNNLSTM:
	$(TIME) python anodecli_SNN.py arson OF-SNNLSTM ./videos/arson011_x264.mp4 ./out/arson011_OF-CNN.pkl

oldanodecli:
	$(TIME) python anodecli.py -m arson -v ./videos/arson001-x264_JT5fU5T6.mp4 -o ./out/arson001-x264_JT5fU5T6.out.pickle

anodeyt:
	$(TIME) python anodeyt.py -m explosion -yt https://www.youtube.com/watch?v=rZXz0sodnvo -o ./out/rZXz0sodnvo.out.pickle
