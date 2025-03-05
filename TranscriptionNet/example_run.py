from Data_process import load_rawdata, datasets_split
from FunDNN.Model import run_model
from FunDNN.preprocessor import load_data, get_dataloader
from GenSAN.preprocessor import GenSAN_preprocessor, get_pre_GECs
from GenSAN.train_function import run_GenSAN_model
from config_parser import TranscriptionNet_Hyperparameters

# load model hyperparameters
config = TranscriptionNet_Hyperparameters()

# load raw data and split into training, validation and test sets
node_feature, RNAi_GECs, OE_GECs, CRISPR_GECs = load_rawdata("example_data/raw_data/")

RNAi_MMScaler = datasets_split(RNAi_GECs, node_feature, "example_data/datasets/RNAi/")
OE_MMScaler = datasets_split(OE_GECs, node_feature, "example_data/datasets/OE/")
CRISPR_MMScaler = datasets_split(CRISPR_GECs, node_feature, "example_data/datasets/CRISPR/")


# RNAi FunDNN model
print("RNAi FunDNN model")
RNAi_feature_train, RNAi_feature_valid, RNAi_feature_test = load_data(config.FunDNN_RNAi_path, "feature_dict.pkl")
RNAi_GECs_train, RNAi_GECs_valid, RNAi_GECs_test = load_data(config.FunDNN_RNAi_path, "GECs_dict.pkl")

RNAi_train_dataloader, RNAi_valid_dataloader = get_dataloader(config.FunDNN_batch_size, RNAi_feature_train,
                                                              RNAi_GECs_train,
                                                              RNAi_feature_valid, RNAi_GECs_valid)

RNAi_pre_GECs = run_model(num_layers=config.FunDNN_layers,
                          hidden_nodes=config.FunDNN_hidden_nodes,
                          activate_func=config.FunDNN_activation_func,
                          dropout_rate=config.FunDNN_dropout_rate,
                          learning_rate=config.FunDNN_learning_rate,
                          epochs=config.FunDNN_epochs,
                          train_dataloader=RNAi_train_dataloader,
                          valid_dataloader=RNAi_valid_dataloader,
                          beta=config.PMSELoss_beta,
                          feature_test=RNAi_feature_test,
                          gecs_test=RNAi_GECs_test,
                          save_path=config.FunDNN_save_path,
                          node_feature=node_feature,
                          name="RNAi")


# OE FunDNN model
print("OE FunDNN model")
OE_feature_train, OE_feature_valid, OE_feature_test = load_data(config.FunDNN_OE_path, "feature_dict.pkl")
OE_GECs_train, OE_GECs_valid, OE_GECs_test = load_data(config.FunDNN_OE_path, "GECs_dict.pkl")

OE_train_dataloader, OE_valid_dataloader = get_dataloader(config.FunDNN_batch_size, OE_feature_train,
                                                          OE_GECs_train,
                                                          OE_feature_valid, OE_GECs_valid)

OE_pre_GECs = run_model(num_layers=config.FunDNN_layers,
                        hidden_nodes=config.FunDNN_hidden_nodes,
                        activate_func=config.FunDNN_activation_func,
                        dropout_rate=config.FunDNN_dropout_rate,
                        learning_rate=config.FunDNN_learning_rate,
                        epochs=config.FunDNN_epochs,
                        train_dataloader=OE_train_dataloader,
                        valid_dataloader=OE_valid_dataloader,
                        beta=config.PMSELoss_beta,
                        feature_test=OE_feature_test,
                        gecs_test=OE_GECs_test,
                        save_path=config.FunDNN_save_path,
                        node_feature=node_feature,
                        name="OE")


# CRISPR FunDNN model
print("CRISPR FunDNN model")
CRISPR_feature_train, CRISPR_feature_valid, CRISPR_feature_test = load_data(config.FunDNN_CRISPR_path,
                                                                            "feature_dict.pkl")
CRISPR_GECs_train, CRISPR_GECs_valid, CRISPR_GECs_test = load_data(config.FunDNN_CRISPR_path, "GECs_dict.pkl")

CRISPR_train_dataloader, CRISPR_valid_dataloader = get_dataloader(config.FunDNN_batch_size, CRISPR_feature_train,
                                                                  CRISPR_GECs_train,
                                                                  CRISPR_feature_valid, CRISPR_GECs_valid)

CRISPR_pre_GECs = run_model(num_layers=config.FunDNN_layers,
                            hidden_nodes=config.FunDNN_hidden_nodes,
                            activate_func=config.FunDNN_activation_func,
                            dropout_rate=config.FunDNN_dropout_rate,
                            learning_rate=config.FunDNN_learning_rate,
                            epochs=config.FunDNN_epochs,
                            train_dataloader=CRISPR_train_dataloader,
                            valid_dataloader=CRISPR_valid_dataloader,
                            beta=config.PMSELoss_beta,
                            feature_test=CRISPR_feature_test,
                            gecs_test=CRISPR_GECs_test,
                            save_path=config.FunDNN_save_path,
                            node_feature=node_feature,
                            name="CRISPR")


# RNAi GenSAN model
print("RNAi GenSAN model")
GenSAN_train, GenSAN_valid, GenSAN_test, OE_combine, CRISPR_combine = GenSAN_preprocessor(true_GECs1=OE_GECs,
                                                                                          true_GECs2=CRISPR_GECs,
                                                                                          predict_GECs1=OE_pre_GECs,
                                                                                          predict_GECs2=CRISPR_pre_GECs,
                                                                                          pre_GECS=RNAi_pre_GECs,
                                                                                          input_path=config.FunDNN_RNAi_path,
                                                                                          file_name="feature_dict.pkl")

GenSAN_train_dataloader, GenSAN_valid_dataloader = get_dataloader(batch_size=config.GenSAN_batch_size,
                                                                  node_train=GenSAN_train,
                                                                  gecs_train=RNAi_GECs_train,
                                                                  node_valid=GenSAN_valid,
                                                                  gecs_valid=RNAi_GECs_valid)

input_matrix = get_pre_GECs(RNAi_pre_GECs, OE_combine, CRISPR_combine)

RNAi_predict_GECs = run_GenSAN_model(blocks=config.GenSAN_blocks,
                                     GECs_dimension=config.GenSAN_GECs_dimension,
                                     hidden_nodes=config.FunDNN_hidden_nodes,
                                     heads=config.GenSAN_heads,
                                     dropout_rate=config.GenSAN_dropout_rate,
                                     recycles=config.GenSAN_recycles,
                                     learning_rate=config.GenSAN_learning_rate,
                                     weight_decay=config.GenSAN_weight_decay,
                                     epochs=config.GenSAN_epochs,
                                     train_dataloader=GenSAN_train_dataloader,
                                     valid_dataloader=GenSAN_valid_dataloader,
                                     beta=config.PMSELoss_beta,
                                     warmup_epoch=config.GenSAN_warmup_epochs,
                                     pre_gecs_test=GenSAN_test,
                                     gecs_test=RNAi_GECs_test,
                                     save_path=config.FunDNN_save_path,
                                     input_matrix=input_matrix,
                                     length=64,
                                     pre_GECs=RNAi_pre_GECs,
                                     scaler=RNAi_MMScaler,
                                     name="RNAi")
