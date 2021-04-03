import numpy as np
import pandas as pd
from keras.utils import to_categorical

# Names of the 42 features
full_features = ["duration", "protocol_type", "service", "flag", "src_bytes",
                 "dst_bytes", "land", "wrong_fragment", "urgent", "hot",
                 "num_failed_logins", "logged_in", "num_compromised",
                 "root_shell", "su_attempted", "num_root",
                 "num_file_creations", "num_shells", "num_access_files",
                 "num_outbound_cmds", "is_host_login", "is_guest_login",
                 "count", "srv_count", "serror_rate", "srv_serror_rate",
                 "rerror_rate", "srv_rerror_rate", "same_srv_rate",
                 "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
                 "dst_host_srv_count", "dst_host_same_srv_rate",
                 "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
                 "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
                 "dst_host_srv_serror_rate", "dst_host_rerror_rate",
                 "dst_host_srv_rerror_rate", "label", "difficulty"]

# Names of all the attacks names (including NSL KDD)
entry_type = {'normal': 'normal',
              'probe': ['ipsweep.', 'nmap.', 'portsweep.',
                        'satan.', 'saint.', 'mscan.'],
              'dos': ['back.', 'land.', 'neptune.', 'pod.', 'smurf.',
                      'teardrop.', 'apache2.', 'udpstorm.', 'processtable.',
                      'mailbomb.'],
              'u2r': ['buffer_overflow.', 'loadmodule.', 'perl.', 'rootkit.',
                      'xterm.', 'ps.', 'sqlattack.'],
              'r2l': ['ftp_write.', 'guess_passwd.', 'imap.', 'multihop.',
                      'phf.', 'spy.', 'warezclient.', 'warezmaster.',
                      'snmpgetattack.', 'named.', 'xlock.', 'xsnoop.',
                      'sendmail.', 'httptunnel.', 'worm.', 'snmpguess.']}

flag_values = ['OTH', 'RSTOS0', 'SF', 'SH',
               'RSTO', 'S2', 'S1', 'REJ', 'S3', 'RSTR', 'S0']

protocol_type_values = ['tcp', 'udp', 'icmp']

symbolic_columns = ["protocol_type", "service", "flag", "land", "logged_in",
                    "is_host_login", "is_guest_login"]


def kdd_encoding(params):
    # ***** DATA PATH *****
    train_data_path = "./data/KDDTrain+.csv"
    test_data_path = "./data/KDDTest+.csv"

    # Load csv data into dataframes and name the feature
    train_df = pd.read_csv(train_data_path, names=full_features)
    test_df = pd.read_csv(test_data_path, names=full_features)

    def process_dataframe(df):
        # Replace connexion type string with an int (also works with NSL)
        df['label'] = df['label'].replace(['normal.', 'normal'], 0)
        for i in range(len(entry_type['probe'])):
            df['label'] = df['label'].replace(
                [entry_type['probe'][i], entry_type['probe'][i][:-1]], 1)
        for i in range(len(entry_type['dos'])):
            df['label'] = df['label'].replace(
                [entry_type['dos'][i], entry_type['dos'][i][:-1]], 2)
        for i in range(len(entry_type['u2r'])):
            df['label'] = df['label'].replace(
                [entry_type['u2r'][i], entry_type['u2r'][i][:-1]], 3)
        for i in range(len(entry_type['r2l'])):
            df['label'] = df['label'].replace(
                [entry_type['r2l'][i], entry_type['r2l'][i][:-1]], 4)

        df = df.drop(columns="difficulty")

        # Assign x (inputs) and y (outputs) of the network
        y = df['label']
        x = df.drop(columns='label')

        def Normalization(df):
            df = df.apply(lambda t: np.log10(t.replace(0, 1))
                          if t.name not in symbolic_columns else t)
            flag_unnested = pd.get_dummies(df.flag, prefix="flag")
            protocol_type_unnested = pd.get_dummies(
                df.protocol_type, prefix="protocol_type")
            service_unnested = pd.get_dummies(
                df.service, prefix="service")

            ohvs = pd.concat([service_unnested, flag_unnested,
                              protocol_type_unnested], axis=1)
            df = pd.concat([df, ohvs], axis=1)
            df = df.drop(columns=['flag', 'protocol_type', 'service'])

            df = df.apply(lambda t: (t-t.min()) / (t.max()-t.min()))

            return df

        x = Normalization(x)

        return x, y

    x_train, Y_train = process_dataframe(train_df)
    x_test, Y_test = process_dataframe(test_df)

    # Apply one-hot encoding to outputs
    y_train = to_categorical(Y_train)
    y_test = to_categorical(Y_test)

    return x_train, x_test, y_train, y_test
