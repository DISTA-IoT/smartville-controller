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
        
        if not self.dashboard_exists('CPU'):
            self.logger.info("Creating new dashboard: CPU ...")
            self.generate_single_dash('CPU','CPU')

        if not self.dashboard_exists('RAM'):
            self.logger.info("Creating new dashboard: RAM...")
            self.generate_single_dash('RAM','RAM')

        if not self.dashboard_exists('RTT'):
            self.logger.info("Creating new dashboard:  RTT ...")
            self.generate_single_dash('RTT','RTT')

        if not self.dashboard_exists('INBOUND'):
            self.logger.info("Creating new dashboard:  INBOUND...")
            self.generate_single_dash('INBOUND','INBOUND')

        if not self.dashboard_exists('OUTBOUND'):
            self.logger.info("Creating new dashboard: OUTBOUND...")
            self.generate_single_dash('OUTBOUND','OUTBOUND')

    
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