














def train_model1():
    # Define callbacks
    early = EarlyStopping(monitor='loss',
        patience= 5,
        verbose= 1,
        mode='auto',
        baseline= None,
        restore_best_weights= True)
    # Get non-augmented data to train
    non_train_generator, non_val_generator, non_test_generator = generate_data(train_dir = train_dir, val_dir = val_dir, test_dir = test_dir)


    # Get augmented data to train (If needed)
    # train_generator, val_generator, _test_generator = generate_data(train_dir = train_dir, val_dir = val_dir, test_dir = test_dir, aug = True)

    # Define model
    non_model1, checkpoint1 = model1(INPUT_SHAPE, N_CLASS)
    non_model1.compile(loss='categorical_crossentropy', optimizer='adam', metrics=["accuracy"])
    print(non_model1.summary())

    # Fit with non-augmented data
    non_model1_history = non_model1.fit(non_train_generator, batch_size= 32, epochs = 20, callbacks=[early, checkpoint1],
                   validation_data = non_val_generator, validation_batch_size = 32)
    return non_model1_history