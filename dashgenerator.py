# This file is part of the "Smartville" project.
# Copyright (c) 2024 University of Insubria
# Licensed under the Apache License 2.0.
# SPDX-License-Identifier: Apache-2.0
# For the full text of the license, visit:
# https://www.apache.org/licenses/LICENSE-2.0

# Smartville is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# Apache License 2.0 for more details.

# You should have received a copy of the Apache License 2.0
# along with Smartville. If not, see <https://www.apache.org/licenses/LICENSE-2.0>.

# Additional licensing information for third-party dependencies
# used in this file can be found in the accompanying `NOTICE` file.
class DashGenerator:
    """
    Classe dedicata all'inserimento delle varie dashboard su Grafana per la visualizzazione 
    interattiva. Ciò viene svolto tramite la libreria che permette la connessione all'host grafana mediante
    la sua chiave api messa a disposizione
    """

    def __init__(self, grafana_connection, logger, max_conn_retries):
        self.grafana_connection = grafana_connection
        self.logger = logger
        self.max_conn_retries = max_conn_retries
        self.generate_all_dashes()


    def generate_all_dashes(self):
        
        if not self.dashboard_exists('CPU_data'):
            self.logger.info("Creating new dashboard: CPU_data ...")
            self.generate_single_dash('CPU_data','CPU (%)')

        if not self.dashboard_exists('RAM_data'):
            self.logger.info("Creating new dashboard: RAM_data...")
            self.generate_single_dash('RAM_data','RAM (GB)')

        if not self.dashboard_exists('PING_data'):
            self.logger.info("Creating new dashboard:  PING_data ...")
            self.generate_single_dash('PING_data','Latenza (ms)')

        if not self.dashboard_exists('INT_data'):
            self.logger.info("Creating new dashboard:  INT_data...")
            self.generate_single_dash('INT_data','Traffico rete in entrata (KBps)')

        if not self.dashboard_exists('ONT_data'):
            self.logger.info("Creating new dashboard: ONT_data creazione in corso...")
            self.generate_single_dash('ONT_data','Traffico rete in uscita (KBps)')

    
    def dashboard_exists(self, dash_UID):
        """
        Metodo di controllo esistenza dashboard
        """
        try:
            # Tentativo di connessione alla dashboard tramite l'UID
            self.grafana_connection.dashboard.get_dashboard(dash_UID)
            return True
        except Exception:
            # Nel caso la connessione non andasse a buon fine, allora significa che la dashboard 
            # non è esistente 
            return False


    def generate_single_dash_with_return(self, dash_UID, dash_name):
        """ 
        Creazione di una nuova dashboard
        """
        # Configurazione dashboard tramite la definizione del suo JSON model
        dashboard_config = {
            "dashboard": {
                "uid": dash_UID,
                "title": dash_name,
                "panels": [],
                "refresh": "5s",
                "time": {
                    "from": "now-15m",
                    "to": "now"
                },
            },
            "overwrite": False
        }

        # Creazione effettiva dashboard 
        try:
            self.grafana_connection.dashboard.update_dashboard(dashboard_config)
            self.logger.info(f"Dashboard with UID '{dash_UID}' created!")
            return True

        except Exception as e:
            self.logger.error(f"Error: {e} try a different username or password")
            return False

    
    def generate_single_dash(self, dash_UID, dash_name):

        # Configurazione dashboard tramite la definizione del suo JSON model
        dashboard_config = {
            "dashboard": {
                "uid": dash_UID,
                "title": dash_name,
                "panels": [],
                "refresh": "5s",
                "time": {
                    "from": "now-15m",
                    "to": "now"
                },
            },
            "overwrite": False
        }

        # Creazione effettiva dashboard 
        self.grafana_connection.dashboard.update_dashboard(dashboard_config)
        self.logger.info(f"Dashboard con UID '{dash_UID}' created!")