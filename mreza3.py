from mreza import get_data, write_history
from modeli_petar import make_model_1

directory_train = "../chest_xray_norm/train"
x_train, y_train = get_data(directory_train)

directory_val = "../chest_xray_norm/val"
x_val, y_val = get_data(directory_val)


model = make_model_1((x_train.shape[1], x_train.shape[2]))

history = model.fit(x_train, y_train, epochs=250, batch_size=0, verbose=1, shuffle=True, validation_data=(x_val,y_val))

model.save("./modeli/petar/model_1_epoha_250.h5")
write_history(history, "./modeli/petar/model_1_epoha_250.json")