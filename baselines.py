{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "e12f0395",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:32.660042Z",
     "iopub.status.busy": "2026-04-02T08:44:32.659187Z",
     "iopub.status.idle": "2026-04-02T08:44:33.417028Z",
     "shell.execute_reply": "2026-04-02T08:44:33.415943Z"
    },
    "papermill": {
     "duration": 0.764642,
     "end_time": "2026-04-02T08:44:33.419316",
     "exception": false,
     "start_time": "2026-04-02T08:44:32.654674",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Cloning into 'DropCount'...\r\n",
      "remote: Enumerating objects: 62, done.\u001b[K\r\n",
      "remote: Counting objects: 100% (62/62), done.\u001b[K\r\n",
      "remote: Compressing objects: 100% (57/57), done.\u001b[K\r\n",
      "remote: Total 62 (delta 21), reused 0 (delta 0), pack-reused 0 (from 0)\u001b[K\r\n",
      "Receiving objects: 100% (62/62), 45.32 KiB | 2.52 MiB/s, done.\r\n",
      "Resolving deltas: 100% (21/21), done.\r\n"
     ]
    }
   ],
   "source": [
    "!git clone https://github.com/3103436559wxc-collab/DropCount.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "b81d6b19",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:33.424650Z",
     "iopub.status.busy": "2026-04-02T08:44:33.424197Z",
     "iopub.status.idle": "2026-04-02T08:44:33.431279Z",
     "shell.execute_reply": "2026-04-02T08:44:33.430379Z"
    },
    "papermill": {
     "duration": 0.011418,
     "end_time": "2026-04-02T08:44:33.432839",
     "exception": false,
     "start_time": "2026-04-02T08:44:33.421421",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/working/DropCount\n"
     ]
    }
   ],
   "source": [
    "%cd DropCount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "9c54a0fe",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:33.437798Z",
     "iopub.status.busy": "2026-04-02T08:44:33.437438Z",
     "iopub.status.idle": "2026-04-02T08:44:38.333222Z",
     "shell.execute_reply": "2026-04-02T08:44:38.332341Z"
    },
    "papermill": {
     "duration": 4.90047,
     "end_time": "2026-04-02T08:44:38.335131",
     "exception": false,
     "start_time": "2026-04-02T08:44:33.434661",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "!pip install -q -r requirements.txt"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "935247bb",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:38.341553Z",
     "iopub.status.busy": "2026-04-02T08:44:38.340633Z",
     "iopub.status.idle": "2026-04-02T08:44:42.901588Z",
     "shell.execute_reply": "2026-04-02T08:44:42.900801Z"
    },
    "papermill": {
     "duration": 4.565939,
     "end_time": "2026-04-02T08:44:42.903285",
     "exception": false,
     "start_time": "2026-04-02T08:44:38.337346",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Torch: 2.9.0+cu126\n",
      "CUDA available: True\n",
      "GPU: Tesla T4\n"
     ]
    }
   ],
   "source": [
    "# Cell 3: verify device\n",
    "import torch\n",
    "print(\"Torch:\", torch.__version__)\n",
    "print(\"CUDA available:\", torch.cuda.is_available())\n",
    "if torch.cuda.is_available():\n",
    "    print(\"GPU:\", torch.cuda.get_device_name(0))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "09a65c7b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:42.908648Z",
     "iopub.status.busy": "2026-04-02T08:44:42.907807Z",
     "iopub.status.idle": "2026-04-02T08:44:42.920707Z",
     "shell.execute_reply": "2026-04-02T08:44:42.920053Z"
    },
    "papermill": {
     "duration": 0.017077,
     "end_time": "2026-04-02T08:44:42.922201",
     "exception": false,
     "start_time": "2026-04-02T08:44:42.905124",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [],
   "source": [
    "%%bash\n",
    "cd /kaggle/working/DropCount\n",
    "\n",
    "cat > configs/one_setting_10k.yaml <<'YAML'\n",
    "simulation:\n",
    "  n_droplets: 10000\n",
    "  true_copy_range: [0, 20000]\n",
    "  copy_sampling_mode: log_uniform_integer\n",
    "  simulation_mode: fixed_total_multinomial\n",
    "  random_seed: 123\n",
    "  false_positive_rate_range: [0.0, 0.05]\n",
    "\n",
    "  train_samples_per_epoch: 20000\n",
    "  val_samples: 1000\n",
    "  test_samples_per_combo: 10000\n",
    "  test_n_droplets: [10000]\n",
    "\n",
    "  copy_bins: [0, 20, 100, 1000, 5000, 10000, 20000]\n",
    "\n",
    "  distributions:\n",
    "    name: truncated_normal\n",
    "    cv: 0.5\n",
    "    train_names: [truncated_normal]\n",
    "    eval_names: [truncated_normal]\n",
    "    cv_values: [0.5]\n",
    "    lognormal_sigma_cap: 2.0\n",
    "    binomial_k: 10\n",
    "    positive_floor: 1.0e-6\n",
    "\n",
    "model:\n",
    "  model_type: perceiver\n",
    "  input_dim: 10\n",
    "  hidden_dim: 128\n",
    "  latent_dim: 128\n",
    "  num_latents: 32\n",
    "  num_heads: 4\n",
    "  num_self_attn_layers: 3\n",
    "  dropout: 0.05\n",
    "  fourier_features: 4\n",
    "  use_rmsnorm: false\n",
    "\n",
    "training:\n",
    "  batch_size: 4\n",
    "  epochs: 300\n",
    "  learning_rate: 2.0e-4\n",
    "  weight_decay: 1.0e-4\n",
    "  grad_clip_norm: 1.0\n",
    "  loss_name: huber_log\n",
    "  linear_loss_weight: 0.05\n",
    "  huber_delta: 0.5\n",
    "  early_stopping_patience: 40\n",
    "  num_workers: 0\n",
    "  amp: true\n",
    "  device: auto\n",
    "\n",
    "output:\n",
    "  run_root: outputs\n",
    "  run_name: one_setting_10k\n",
    "  save_validation_dataset: true\n",
    "\n",
    "baselines:\n",
    "  max_copy_cap: 1000000.0\n",
    "  eps: 1.0e-12\n",
    "  mle_search_upper_multiplier: 20.0\n",
    "YAML"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "c43fdb6e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T08:44:42.927225Z",
     "iopub.status.busy": "2026-04-02T08:44:42.926678Z",
     "iopub.status.idle": "2026-04-02T12:31:21.560086Z",
     "shell.execute_reply": "2026-04-02T12:31:21.559306Z"
    },
    "papermill": {
     "duration": 13598.638089,
     "end_time": "2026-04-02T12:31:21.562153",
     "exception": false,
     "start_time": "2026-04-02T08:44:42.924064",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\"epoch\": 1, \"train_loss\": 0.4521828709143214, \"val_loss\": 0.274814913418144, \"learning_rate\": 0.00019999451693655123}\r\n",
      "{\"epoch\": 2, \"train_loss\": 0.3266157824437134, \"val_loss\": 0.23589157381653786, \"learning_rate\": 0.00019997806834748456}\r\n",
      "{\"epoch\": 3, \"train_loss\": 0.28574492006367075, \"val_loss\": 0.17670295664295554, \"learning_rate\": 0.00019995065603657316}\r\n",
      "{\"epoch\": 4, \"train_loss\": 0.27503392522088255, \"val_loss\": 0.2517247272506356, \"learning_rate\": 0.00019991228300988585}\r\n",
      "{\"epoch\": 5, \"train_loss\": 0.2548278384491801, \"val_loss\": 0.23932002464681865, \"learning_rate\": 0.0001998629534754574}\r\n",
      "{\"epoch\": 6, \"train_loss\": 0.25529914560159667, \"val_loss\": 0.15886277723871173, \"learning_rate\": 0.00019980267284282717}\r\n",
      "{\"epoch\": 7, \"train_loss\": 0.24350289239212872, \"val_loss\": 0.30469846511632204, \"learning_rate\": 0.00019973144772244582}\r\n",
      "{\"epoch\": 8, \"train_loss\": 0.2354667067591101, \"val_loss\": 0.17863870010524988, \"learning_rate\": 0.00019964928592495045}\r\n",
      "{\"epoch\": 9, \"train_loss\": 0.23234171183034777, \"val_loss\": 0.16524089569645004, \"learning_rate\": 0.000199556196460308}\r\n",
      "{\"epoch\": 10, \"train_loss\": 0.2273553145455662, \"val_loss\": 0.17354169193655253, \"learning_rate\": 0.00019945218953682734}\r\n",
      "{\"epoch\": 11, \"train_loss\": 0.22408604164093268, \"val_loss\": 0.276158053457737, \"learning_rate\": 0.00019933727656003966}\r\n",
      "{\"epoch\": 12, \"train_loss\": 0.20951941112340428, \"val_loss\": 0.19064217788726093, \"learning_rate\": 0.00019921147013144782}\r\n",
      "{\"epoch\": 13, \"train_loss\": 0.21133551993994043, \"val_loss\": 0.1754664206802845, \"learning_rate\": 0.0001990747840471444}\r\n",
      "{\"epoch\": 14, \"train_loss\": 0.20662069846296216, \"val_loss\": 0.12829474177071826, \"learning_rate\": 0.00019892723329629887}\r\n",
      "{\"epoch\": 15, \"train_loss\": 0.1951021085887216, \"val_loss\": 0.14756408451870084, \"learning_rate\": 0.0001987688340595138}\r\n",
      "{\"epoch\": 16, \"train_loss\": 0.19496003557173536, \"val_loss\": 0.20196952520823105, \"learning_rate\": 0.0001985996037070505}\r\n",
      "{\"epoch\": 17, \"train_loss\": 0.19847896094499157, \"val_loss\": 0.1577731181792915, \"learning_rate\": 0.00019841956079692417}\r\n",
      "{\"epoch\": 18, \"train_loss\": 0.187986966614821, \"val_loss\": 0.12427087821252644, \"learning_rate\": 0.00019822872507286888}\r\n",
      "{\"epoch\": 19, \"train_loss\": 0.18697228097741025, \"val_loss\": 0.13290457397140562, \"learning_rate\": 0.00019802711746217218}\r\n",
      "{\"epoch\": 20, \"train_loss\": 0.18564289515842683, \"val_loss\": 0.21735172070749104, \"learning_rate\": 0.00019781476007338055}\r\n",
      "{\"epoch\": 21, \"train_loss\": 0.18649941730047576, \"val_loss\": 0.1327783241830766, \"learning_rate\": 0.0001975916761938747}\r\n",
      "{\"epoch\": 22, \"train_loss\": 0.19023012581325602, \"val_loss\": 0.12498427935084327, \"learning_rate\": 0.000197357890287316}\r\n",
      "{\"epoch\": 23, \"train_loss\": 0.17703090481571854, \"val_loss\": 0.16132961053703912, \"learning_rate\": 0.0001971134279909636}\r\n",
      "{\"epoch\": 24, \"train_loss\": 0.17934677760704654, \"val_loss\": 0.1330509847458452, \"learning_rate\": 0.00019685831611286308}\r\n",
      "{\"epoch\": 25, \"train_loss\": 0.17494950307668186, \"val_loss\": 0.2214525236338377, \"learning_rate\": 0.00019659258262890678}\r\n",
      "{\"epoch\": 26, \"train_loss\": 0.1713465764710214, \"val_loss\": 0.14881206757202745, \"learning_rate\": 0.00019631625667976578}\r\n",
      "{\"epoch\": 27, \"train_loss\": 0.17761583974855022, \"val_loss\": 0.18596193746756762, \"learning_rate\": 0.00019602936856769426}\r\n",
      "{\"epoch\": 28, \"train_loss\": 0.17864738642238082, \"val_loss\": 0.14706110041029752, \"learning_rate\": 0.00019573194975320668}\r\n",
      "{\"epoch\": 29, \"train_loss\": 0.18041827842271888, \"val_loss\": 0.13450669422373177, \"learning_rate\": 0.00019542403285162765}\r\n",
      "{\"epoch\": 30, \"train_loss\": 0.1718716643325286, \"val_loss\": 0.21523535507917405, \"learning_rate\": 0.00019510565162951534}\r\n",
      "{\"epoch\": 31, \"train_loss\": 0.17249365024159197, \"val_loss\": 0.23354388366639614, \"learning_rate\": 0.00019477684100095856}\r\n",
      "{\"epoch\": 32, \"train_loss\": 0.16615622670094016, \"val_loss\": 0.13497090366529302, \"learning_rate\": 0.0001944376370237481}\r\n",
      "{\"epoch\": 33, \"train_loss\": 0.16922464923874941, \"val_loss\": 0.18339896123297514, \"learning_rate\": 0.0001940880768954225}\r\n",
      "{\"epoch\": 34, \"train_loss\": 0.16778740372860337, \"val_loss\": 0.25460032432619484, \"learning_rate\": 0.0001937281989491891}\r\n",
      "{\"epoch\": 35, \"train_loss\": 0.1677471365195699, \"val_loss\": 0.12814179687271826, \"learning_rate\": 0.0001933580426497201}\r\n",
      "{\"epoch\": 36, \"train_loss\": 0.167143053664919, \"val_loss\": 0.18095229854807257, \"learning_rate\": 0.00019297764858882508}\r\n",
      "{\"epoch\": 37, \"train_loss\": 0.16294585983420257, \"val_loss\": 0.3086252562517766, \"learning_rate\": 0.00019258705848099945}\r\n",
      "{\"epoch\": 38, \"train_loss\": 0.16447667760320472, \"val_loss\": 0.14808838198054583, \"learning_rate\": 0.00019218631515885003}\r\n",
      "{\"epoch\": 39, \"train_loss\": 0.16317146087735893, \"val_loss\": 0.13523838280700148, \"learning_rate\": 0.0001917754625683981}\r\n",
      "{\"epoch\": 40, \"train_loss\": 0.16388091036854313, \"val_loss\": 0.17106330267246814, \"learning_rate\": 0.00019135454576426006}\r\n",
      "{\"epoch\": 41, \"train_loss\": 0.16390864851402584, \"val_loss\": 0.2052241612058133, \"learning_rate\": 0.00019092361090470685}\r\n",
      "{\"epoch\": 42, \"train_loss\": 0.16594243241841905, \"val_loss\": 0.25650724701583383, \"learning_rate\": 0.00019048270524660196}\r\n",
      "{\"epoch\": 43, \"train_loss\": 0.1635783179932041, \"val_loss\": 0.11218159660382662, \"learning_rate\": 0.00019003187714021935}\r\n",
      "{\"epoch\": 44, \"train_loss\": 0.16143941520156804, \"val_loss\": 0.127656230263412, \"learning_rate\": 0.00018957117602394128}\r\n",
      "{\"epoch\": 45, \"train_loss\": 0.16314012847428674, \"val_loss\": 0.11668687013932504, \"learning_rate\": 0.00018910065241883677}\r\n",
      "{\"epoch\": 46, \"train_loss\": 0.15853563645326066, \"val_loss\": 0.1644169094413519, \"learning_rate\": 0.00018862035792312145}\r\n",
      "{\"epoch\": 47, \"train_loss\": 0.16236529023775365, \"val_loss\": 0.1315291581712663, \"learning_rate\": 0.00018813034520649919}\r\n",
      "{\"epoch\": 48, \"train_loss\": 0.16259741493309848, \"val_loss\": 0.1338863903619349, \"learning_rate\": 0.00018763066800438633}\r\n",
      "{\"epoch\": 49, \"train_loss\": 0.1554415686528082, \"val_loss\": 0.13831003409996628, \"learning_rate\": 0.00018712138111201895}\r\n",
      "{\"epoch\": 50, \"train_loss\": 0.15837083071572705, \"val_loss\": 0.12150808652862906, \"learning_rate\": 0.00018660254037844386}\r\n",
      "{\"epoch\": 51, \"train_loss\": 0.15841135088936426, \"val_loss\": 0.14219894714653492, \"learning_rate\": 0.00018607420270039436}\r\n",
      "{\"epoch\": 52, \"train_loss\": 0.1590291886426974, \"val_loss\": 0.19643612439185382, \"learning_rate\": 0.00018553642601605065}\r\n",
      "{\"epoch\": 53, \"train_loss\": 0.153851552635571, \"val_loss\": 0.1544245765833184, \"learning_rate\": 0.0001849892692986864}\r\n",
      "{\"epoch\": 54, \"train_loss\": 0.15587908456206787, \"val_loss\": 0.1252773699518293, \"learning_rate\": 0.00018443279255020152}\r\n",
      "{\"epoch\": 55, \"train_loss\": 0.15251386406361125, \"val_loss\": 0.11478786559868603, \"learning_rate\": 0.00018386705679454242}\r\n",
      "{\"epoch\": 56, \"train_loss\": 0.15500949921751161, \"val_loss\": 0.13621619086526335, \"learning_rate\": 0.00018329212407100997}\r\n",
      "{\"epoch\": 57, \"train_loss\": 0.1485602028847672, \"val_loss\": 0.19185743165109306, \"learning_rate\": 0.0001827080574274562}\r\n",
      "{\"epoch\": 58, \"train_loss\": 0.15137409101913218, \"val_loss\": 0.140019178181421, \"learning_rate\": 0.00018211492091337042}\r\n",
      "{\"epoch\": 59, \"train_loss\": 0.15497724654267075, \"val_loss\": 0.11084023963892832, \"learning_rate\": 0.00018151277957285543}\r\n",
      "{\"epoch\": 60, \"train_loss\": 0.15386204856960103, \"val_loss\": 0.1668299533687532, \"learning_rate\": 0.00018090169943749476}\r\n",
      "{\"epoch\": 61, \"train_loss\": 0.15537235879036598, \"val_loss\": 0.115899099992821, \"learning_rate\": 0.00018028174751911146}\r\n",
      "{\"epoch\": 62, \"train_loss\": 0.15375259984084405, \"val_loss\": 0.14160372404754162, \"learning_rate\": 0.00017965299180241963}\r\n",
      "{\"epoch\": 63, \"train_loss\": 0.15375055289901793, \"val_loss\": 0.18424634618964048, \"learning_rate\": 0.000179015501237569}\r\n",
      "{\"epoch\": 64, \"train_loss\": 0.15005329305296763, \"val_loss\": 0.14383746384829282, \"learning_rate\": 0.00017836934573258397}\r\n",
      "{\"epoch\": 65, \"train_loss\": 0.14905303813901263, \"val_loss\": 0.11838851110311226, \"learning_rate\": 0.00017771459614569708}\r\n",
      "{\"epoch\": 66, \"train_loss\": 0.15137537051256514, \"val_loss\": 0.1271500095874071, \"learning_rate\": 0.00017705132427757892}\r\n",
      "{\"epoch\": 67, \"train_loss\": 0.14838955580189359, \"val_loss\": 0.12974237385299056, \"learning_rate\": 0.00017637960286346423}\r\n",
      "{\"epoch\": 68, \"train_loss\": 0.1513765765658114, \"val_loss\": 0.13389161349833012, \"learning_rate\": 0.00017569950556517563}\r\n",
      "{\"epoch\": 69, \"train_loss\": 0.15129523433281575, \"val_loss\": 0.12371402190625667, \"learning_rate\": 0.00017501110696304596}\r\n",
      "{\"epoch\": 70, \"train_loss\": 0.1470726549192099, \"val_loss\": 0.12856921931076795, \"learning_rate\": 0.00017431448254773944}\r\n",
      "{\"epoch\": 71, \"train_loss\": 0.14956099793494212, \"val_loss\": 0.17493934474140405, \"learning_rate\": 0.00017360970871197344}\r\n",
      "{\"epoch\": 72, \"train_loss\": 0.14949242382324301, \"val_loss\": 0.12410278942249715, \"learning_rate\": 0.00017289686274214118}\r\n",
      "{\"epoch\": 73, \"train_loss\": 0.1482014861012809, \"val_loss\": 0.13460191669873894, \"learning_rate\": 0.00017217602280983623}\r\n",
      "{\"epoch\": 74, \"train_loss\": 0.14372776185579134, \"val_loss\": 0.16654495434556157, \"learning_rate\": 0.00017144726796328034}\r\n",
      "{\"epoch\": 75, \"train_loss\": 0.15022576006988528, \"val_loss\": 0.15642439108714462, \"learning_rate\": 0.00017071067811865476}\r\n",
      "{\"epoch\": 76, \"train_loss\": 0.1440832190090092, \"val_loss\": 0.12787220891052858, \"learning_rate\": 0.00016996633405133658}\r\n",
      "{\"epoch\": 77, \"train_loss\": 0.1454563260976458, \"val_loss\": 0.16843229781882837, \"learning_rate\": 0.0001692143173870407}\r\n",
      "{\"epoch\": 78, \"train_loss\": 0.1467903039192548, \"val_loss\": 0.12079415063560009, \"learning_rate\": 0.0001684547105928689}\r\n",
      "{\"epoch\": 79, \"train_loss\": 0.1422368512449204, \"val_loss\": 0.1568525336929597, \"learning_rate\": 0.0001676875969682661}\r\n",
      "{\"epoch\": 80, \"train_loss\": 0.14245858046350768, \"val_loss\": 0.12663114805426448, \"learning_rate\": 0.00016691306063588586}\r\n",
      "{\"epoch\": 81, \"train_loss\": 0.14421222780877724, \"val_loss\": 0.17006290847714992, \"learning_rate\": 0.00016613118653236524}\r\n",
      "{\"epoch\": 82, \"train_loss\": 0.1436158373530023, \"val_loss\": 0.11860605263710022, \"learning_rate\": 0.00016534206039901063}\r\n",
      "{\"epoch\": 83, \"train_loss\": 0.14377660029563122, \"val_loss\": 0.11615672152861953, \"learning_rate\": 0.00016454576877239513}\r\n",
      "{\"epoch\": 84, \"train_loss\": 0.14398617752005813, \"val_loss\": 0.11708745006844401, \"learning_rate\": 0.00016374239897486904}\r\n",
      "{\"epoch\": 85, \"train_loss\": 0.14532527654494626, \"val_loss\": 0.11084569830121473, \"learning_rate\": 0.0001629320391049838}\r\n",
      "{\"epoch\": 86, \"train_loss\": 0.13970687220999972, \"val_loss\": 0.1146469055567868, \"learning_rate\": 0.0001621147780278311}\r\n",
      "{\"epoch\": 87, \"train_loss\": 0.14353948802321684, \"val_loss\": 0.14736541313678025, \"learning_rate\": 0.00016129070536529771}\r\n",
      "{\"epoch\": 88, \"train_loss\": 0.14138068294625264, \"val_loss\": 0.1289343647658825, \"learning_rate\": 0.00016045991148623756}\r\n",
      "{\"epoch\": 89, \"train_loss\": 0.14622313858494163, \"val_loss\": 0.15137213064264507, \"learning_rate\": 0.00015962248749656164}\r\n",
      "{\"epoch\": 90, \"train_loss\": 0.1471933376307483, \"val_loss\": 0.11708343356614932, \"learning_rate\": 0.00015877852522924737}\r\n",
      "{\"epoch\": 91, \"train_loss\": 0.14538260713806375, \"val_loss\": 0.14957571381051094, \"learning_rate\": 0.00015792811723426794}\r\n",
      "{\"epoch\": 92, \"train_loss\": 0.1450235928798327, \"val_loss\": 0.11492616801522672, \"learning_rate\": 0.00015707135676844327}\r\n",
      "{\"epoch\": 93, \"train_loss\": 0.1459498493606923, \"val_loss\": 0.11319258539751172, \"learning_rate\": 0.00015620833778521315}\r\n",
      "{\"epoch\": 94, \"train_loss\": 0.14111056733691366, \"val_loss\": 0.13120081198215486, \"learning_rate\": 0.00015533915492433448}\r\n",
      "{\"epoch\": 95, \"train_loss\": 0.1429721330411965, \"val_loss\": 0.11783022607304156, \"learning_rate\": 0.00015446390350150278}\r\n",
      "{\"epoch\": 96, \"train_loss\": 0.14153922006346983, \"val_loss\": 0.11502213120553642, \"learning_rate\": 0.00015358267949789974}\r\n",
      "{\"epoch\": 97, \"train_loss\": 0.14196060676609631, \"val_loss\": 0.17985493782162668, \"learning_rate\": 0.00015269557954966786}\r\n",
      "{\"epoch\": 98, \"train_loss\": 0.14238881634690334, \"val_loss\": 0.11086658827075735, \"learning_rate\": 0.0001518027009373131}\r\n",
      "{\"epoch\": 99, \"train_loss\": 0.1438520665436052, \"val_loss\": 0.12082326727500185, \"learning_rate\": 0.00015090414157503722}\r\n",
      "Early stopping at epoch 99; best epoch was 59\r\n",
      "Saved best checkpoint to outputs/one_setting_10k/best_model.pt\r\n"
     ]
    }
   ],
   "source": [
    "!python train.py --config configs/one_setting_10k.yaml --run-name one_setting_10k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "e62ca04a",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T12:31:21.574405Z",
     "iopub.status.busy": "2026-04-02T12:31:21.574010Z",
     "iopub.status.idle": "2026-04-02T12:31:26.555516Z",
     "shell.execute_reply": "2026-04-02T12:31:26.554540Z"
    },
    "papermill": {
     "duration": 4.989815,
     "end_time": "2026-04-02T12:31:26.557425",
     "exception": false,
     "start_time": "2026-04-02T12:31:21.567610",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Traceback (most recent call last):\r\n",
      "  File \"/kaggle/working/DropCount/evaluate.py\", line 15, in <module>\r\n",
      "    from baselines import naive_equal_volume_estimate, volume_aware_mle_estimate\r\n",
      "  File \"/kaggle/working/DropCount/baselines.py\", line 17, in <module>\r\n",
      "    \"exception\": false,\r\n",
      "                 ^^^^^\r\n",
      "NameError: name 'false' is not defined. Did you mean: 'False'?\r\n"
     ]
    }
   ],
   "source": [
    "!python evaluate.py --config configs/one_setting_10k.yaml --run-dir outputs/one_setting_10k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7a442bc6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T12:31:26.569795Z",
     "iopub.status.busy": "2026-04-02T12:31:26.569523Z",
     "iopub.status.idle": "2026-04-02T12:31:26.694696Z",
     "shell.execute_reply": "2026-04-02T12:31:26.694005Z"
    },
    "papermill": {
     "duration": 0.133285,
     "end_time": "2026-04-02T12:31:26.696205",
     "exception": false,
     "start_time": "2026-04-02T12:31:26.562920",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total 171076\r\n",
      "drwxr-xr-x 2 root root      4096 Apr  2 12:31 .\r\n",
      "drwxr-xr-x 3 root root      4096 Apr  2 08:44 ..\r\n",
      "-rw-r--r-- 1 root root   3358709 Apr  2 11:00 best_model.pt\r\n",
      "-rw-r--r-- 1 root root      1268 Apr  2 12:31 config.yaml\r\n",
      "-rw-r--r-- 1 root root        77 Apr  2 12:31 summary.json\r\n",
      "-rw-r--r-- 1 root root     92028 Apr  2 12:31 training_curves.png\r\n",
      "-rw-r--r-- 1 root root      6480 Apr  2 12:31 training_history.csv\r\n",
      "-rw-r--r-- 1 root root 171703801 Apr  2 12:31 validation_dataset.pt\r\n"
     ]
    }
   ],
   "source": [
    "!ls -la /kaggle/working/DropCount/outputs/one_setting_10k"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "nvidiaTeslaT4",
   "dataSources": [],
   "dockerImageVersionId": 31286,
   "isGpuEnabled": true,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.12"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 13617.653019,
   "end_time": "2026-04-02T12:31:27.521967",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-04-02T08:44:29.868948",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
