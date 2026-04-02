{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7ce670d2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:09:46.667456Z",
     "iopub.status.busy": "2026-04-02T13:09:46.667099Z",
     "iopub.status.idle": "2026-04-02T13:09:47.775206Z",
     "shell.execute_reply": "2026-04-02T13:09:47.774219Z"
    },
    "papermill": {
     "duration": 1.115071,
     "end_time": "2026-04-02T13:09:47.777163",
     "exception": false,
     "start_time": "2026-04-02T13:09:46.662092",
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
      "remote: Enumerating objects: 47, done.\u001b[K\r\n",
      "remote: Counting objects: 100% (47/47), done.\u001b[K\r\n",
      "remote: Compressing objects: 100% (42/42), done.\u001b[K\r\n",
      "remote: Total 47 (delta 14), reused 0 (delta 0), pack-reused 0 (from 0)\u001b[K\r\n",
      "Receiving objects: 100% (47/47), 29.96 KiB | 4.28 MiB/s, done.\r\n",
      "Resolving deltas: 100% (14/14), done.\r\n"
     ]
    }
   ],
   "source": [
    "!git clone https://github.com/xchlai/DropCount.git"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "1134331c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:09:47.783495Z",
     "iopub.status.busy": "2026-04-02T13:09:47.782701Z",
     "iopub.status.idle": "2026-04-02T13:09:47.790133Z",
     "shell.execute_reply": "2026-04-02T13:09:47.789381Z"
    },
    "papermill": {
     "duration": 0.012334,
     "end_time": "2026-04-02T13:09:47.791743",
     "exception": false,
     "start_time": "2026-04-02T13:09:47.779409",
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
   "id": "ea2c358f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:09:47.797418Z",
     "iopub.status.busy": "2026-04-02T13:09:47.796725Z",
     "iopub.status.idle": "2026-04-02T13:09:53.747453Z",
     "shell.execute_reply": "2026-04-02T13:09:53.746557Z"
    },
    "papermill": {
     "duration": 5.955719,
     "end_time": "2026-04-02T13:09:53.749420",
     "exception": false,
     "start_time": "2026-04-02T13:09:47.793701",
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
   "id": "1919f764",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:09:53.755130Z",
     "iopub.status.busy": "2026-04-02T13:09:53.754455Z",
     "iopub.status.idle": "2026-04-02T13:10:00.807606Z",
     "shell.execute_reply": "2026-04-02T13:10:00.806682Z"
    },
    "papermill": {
     "duration": 7.057796,
     "end_time": "2026-04-02T13:10:00.809123",
     "exception": false,
     "start_time": "2026-04-02T13:09:53.751327",
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
   "id": "528976b3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:10:00.814068Z",
     "iopub.status.busy": "2026-04-02T13:10:00.813723Z",
     "iopub.status.idle": "2026-04-02T13:10:00.827434Z",
     "shell.execute_reply": "2026-04-02T13:10:00.826735Z"
    },
    "papermill": {
     "duration": 0.017923,
     "end_time": "2026-04-02T13:10:00.828870",
     "exception": false,
     "start_time": "2026-04-02T13:10:00.810947",
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
   "id": "62dcdf1e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T13:10:00.832981Z",
     "iopub.status.busy": "2026-04-02T13:10:00.832754Z",
     "iopub.status.idle": "2026-04-02T19:16:28.635699Z",
     "shell.execute_reply": "2026-04-02T19:16:28.634972Z"
    },
    "papermill": {
     "duration": 21987.807085,
     "end_time": "2026-04-02T19:16:28.637569",
     "exception": false,
     "start_time": "2026-04-02T13:10:00.830484",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\"epoch\": 1, \"train_loss\": 1.08376745541282, \"val_loss\": 0.32891405749693514, \"learning_rate\": 0.00019999451693655123}\r\n",
      "{\"epoch\": 2, \"train_loss\": 0.32750929253101346, \"val_loss\": 0.20301469060964883, \"learning_rate\": 0.00019997806834748456}\r\n",
      "{\"epoch\": 3, \"train_loss\": 0.28623482854100873, \"val_loss\": 0.20041439781757073, \"learning_rate\": 0.00019995065603657316}\r\n",
      "{\"epoch\": 4, \"train_loss\": 0.2557013525303453, \"val_loss\": 0.3782676893472672, \"learning_rate\": 0.00019991228300988585}\r\n",
      "{\"epoch\": 5, \"train_loss\": 0.24638804688155652, \"val_loss\": 0.16424935915553943, \"learning_rate\": 0.0001998629534754574}\r\n",
      "{\"epoch\": 6, \"train_loss\": 0.23589816766101868, \"val_loss\": 0.19670644737221302, \"learning_rate\": 0.00019980267284282717}\r\n",
      "{\"epoch\": 7, \"train_loss\": 0.2268315720865503, \"val_loss\": 0.17274863613350316, \"learning_rate\": 0.00019973144772244582}\r\n",
      "{\"epoch\": 8, \"train_loss\": 0.22056313504669817, \"val_loss\": 0.3585368516240269, \"learning_rate\": 0.00019964928592495045}\r\n",
      "{\"epoch\": 9, \"train_loss\": 0.22031197547572665, \"val_loss\": 0.1427149132406339, \"learning_rate\": 0.000199556196460308}\r\n",
      "{\"epoch\": 10, \"train_loss\": 0.21240102390979881, \"val_loss\": 0.19340656445175408, \"learning_rate\": 0.00019945218953682734}\r\n",
      "{\"epoch\": 11, \"train_loss\": 0.20763311364182738, \"val_loss\": 0.16641968177631497, \"learning_rate\": 0.00019933727656003966}\r\n",
      "{\"epoch\": 12, \"train_loss\": 0.20356176245650276, \"val_loss\": 0.1561283257380128, \"learning_rate\": 0.00019921147013144782}\r\n",
      "{\"epoch\": 13, \"train_loss\": 0.202770682184631, \"val_loss\": 0.27371371384337545, \"learning_rate\": 0.0001990747840471444}\r\n",
      "{\"epoch\": 14, \"train_loss\": 0.20109266616862734, \"val_loss\": 0.14612897824728863, \"learning_rate\": 0.00019892723329629887}\r\n",
      "{\"epoch\": 15, \"train_loss\": 0.1884017945791129, \"val_loss\": 0.15986912460066377, \"learning_rate\": 0.0001987688340595138}\r\n",
      "{\"epoch\": 16, \"train_loss\": 0.19428688199052122, \"val_loss\": 0.22144735729135573, \"learning_rate\": 0.0001985996037070505}\r\n",
      "{\"epoch\": 17, \"train_loss\": 0.19260427982129621, \"val_loss\": 0.1422213578671217, \"learning_rate\": 0.00019841956079692417}\r\n",
      "{\"epoch\": 18, \"train_loss\": 0.18180765267247334, \"val_loss\": 0.13609118249360472, \"learning_rate\": 0.00019822872507286888}\r\n",
      "{\"epoch\": 19, \"train_loss\": 0.18865160914883017, \"val_loss\": 0.33962852039933206, \"learning_rate\": 0.00019802711746217218}\r\n",
      "{\"epoch\": 20, \"train_loss\": 0.18383071104856208, \"val_loss\": 0.19472838890226557, \"learning_rate\": 0.00019781476007338055}\r\n",
      "{\"epoch\": 21, \"train_loss\": 0.18351419045841322, \"val_loss\": 0.16898114679381251, \"learning_rate\": 0.0001975916761938747}\r\n",
      "{\"epoch\": 22, \"train_loss\": 0.188677630668669, \"val_loss\": 0.14241308371908962, \"learning_rate\": 0.000197357890287316}\r\n",
      "{\"epoch\": 23, \"train_loss\": 0.1775878036889713, \"val_loss\": 0.17267712364858018, \"learning_rate\": 0.0001971134279909636}\r\n",
      "{\"epoch\": 24, \"train_loss\": 0.17935594097743743, \"val_loss\": 0.1550272883339785, \"learning_rate\": 0.00019685831611286308}\r\n",
      "{\"epoch\": 25, \"train_loss\": 0.17588394230641424, \"val_loss\": 0.14321047024801373, \"learning_rate\": 0.00019659258262890678}\r\n",
      "{\"epoch\": 26, \"train_loss\": 0.17175792994552758, \"val_loss\": 0.14515535549400374, \"learning_rate\": 0.00019631625667976578}\r\n",
      "{\"epoch\": 27, \"train_loss\": 0.17906424624403008, \"val_loss\": 0.13654129358474165, \"learning_rate\": 0.00019602936856769426}\r\n",
      "{\"epoch\": 28, \"train_loss\": 0.17165262312663254, \"val_loss\": 0.13080492437258362, \"learning_rate\": 0.00019573194975320668}\r\n",
      "{\"epoch\": 29, \"train_loss\": 0.1782981375998468, \"val_loss\": 0.19714597940072418, \"learning_rate\": 0.00019542403285162765}\r\n",
      "{\"epoch\": 30, \"train_loss\": 0.1720598332769703, \"val_loss\": 0.13818754873611033, \"learning_rate\": 0.00019510565162951534}\r\n",
      "{\"epoch\": 31, \"train_loss\": 0.17152094284486957, \"val_loss\": 0.15883769223093985, \"learning_rate\": 0.00019477684100095856}\r\n",
      "{\"epoch\": 32, \"train_loss\": 0.16962388232573866, \"val_loss\": 0.16813708748389036, \"learning_rate\": 0.0001944376370237481}\r\n",
      "{\"epoch\": 33, \"train_loss\": 0.16884205837019253, \"val_loss\": 0.17248110560979693, \"learning_rate\": 0.0001940880768954225}\r\n",
      "{\"epoch\": 34, \"train_loss\": 0.16837875011779835, \"val_loss\": 0.12475110026309266, \"learning_rate\": 0.0001937281989491891}\r\n",
      "{\"epoch\": 35, \"train_loss\": 0.1637747479812475, \"val_loss\": 0.14600802527368067, \"learning_rate\": 0.0001933580426497201}\r\n",
      "{\"epoch\": 36, \"train_loss\": 0.16497215623280498, \"val_loss\": 0.15579666874650866, \"learning_rate\": 0.00019297764858882508}\r\n",
      "{\"epoch\": 37, \"train_loss\": 0.16631323981531895, \"val_loss\": 0.22239421535795553, \"learning_rate\": 0.00019258705848099945}\r\n",
      "{\"epoch\": 38, \"train_loss\": 0.16196881492403337, \"val_loss\": 0.14330760402604936, \"learning_rate\": 0.00019218631515885003}\r\n",
      "{\"epoch\": 39, \"train_loss\": 0.16485488590099848, \"val_loss\": 0.14025261809071526, \"learning_rate\": 0.0001917754625683981}\r\n",
      "{\"epoch\": 40, \"train_loss\": 0.16635392129070825, \"val_loss\": 0.15255882005766033, \"learning_rate\": 0.00019135454576426006}\r\n",
      "{\"epoch\": 41, \"train_loss\": 0.1641274876334006, \"val_loss\": 0.28525987377483397, \"learning_rate\": 0.00019092361090470685}\r\n",
      "{\"epoch\": 42, \"train_loss\": 0.16544189331624656, \"val_loss\": 0.1820462515396066, \"learning_rate\": 0.00019048270524660196}\r\n",
      "{\"epoch\": 43, \"train_loss\": 0.16382146610699128, \"val_loss\": 0.1759616977693513, \"learning_rate\": 0.00019003187714021935}\r\n",
      "{\"epoch\": 44, \"train_loss\": 0.1619201300400542, \"val_loss\": 0.1238354889396578, \"learning_rate\": 0.00018957117602394128}\r\n",
      "{\"epoch\": 45, \"train_loss\": 0.1624696143962443, \"val_loss\": 0.22129255682229995, \"learning_rate\": 0.00018910065241883677}\r\n",
      "{\"epoch\": 46, \"train_loss\": 0.15848780738722998, \"val_loss\": 0.23204761140141636, \"learning_rate\": 0.00018862035792312145}\r\n",
      "{\"epoch\": 47, \"train_loss\": 0.1613065397974453, \"val_loss\": 0.17496793546155096, \"learning_rate\": 0.00018813034520649919}\r\n",
      "{\"epoch\": 48, \"train_loss\": 0.1635523127773311, \"val_loss\": 0.12052669758396223, \"learning_rate\": 0.00018763066800438633}\r\n",
      "{\"epoch\": 49, \"train_loss\": 0.16039103261849377, \"val_loss\": 0.12715190538857132, \"learning_rate\": 0.00018712138111201895}\r\n",
      "{\"epoch\": 50, \"train_loss\": 0.16072149388552642, \"val_loss\": 0.14215730416681618, \"learning_rate\": 0.00018660254037844386}\r\n",
      "{\"epoch\": 51, \"train_loss\": 0.15909210876543076, \"val_loss\": 0.18994067168608308, \"learning_rate\": 0.00018607420270039436}\r\n",
      "{\"epoch\": 52, \"train_loss\": 0.1569581135165412, \"val_loss\": 0.11733248793985694, \"learning_rate\": 0.00018553642601605065}\r\n",
      "{\"epoch\": 53, \"train_loss\": 0.15229428628128952, \"val_loss\": 0.11543109895847738, \"learning_rate\": 0.0001849892692986864}\r\n",
      "{\"epoch\": 54, \"train_loss\": 0.15551747694937512, \"val_loss\": 0.11826515545137227, \"learning_rate\": 0.00018443279255020152}\r\n",
      "{\"epoch\": 55, \"train_loss\": 0.1559331937473733, \"val_loss\": 0.12349641890451311, \"learning_rate\": 0.00018386705679454242}\r\n",
      "{\"epoch\": 56, \"train_loss\": 0.1564298631537473, \"val_loss\": 0.24584023783681913, \"learning_rate\": 0.00018329212407100997}\r\n",
      "{\"epoch\": 57, \"train_loss\": 0.14919061289005914, \"val_loss\": 0.2488919642828405, \"learning_rate\": 0.0001827080574274562}\r\n",
      "{\"epoch\": 58, \"train_loss\": 0.15443298999629915, \"val_loss\": 0.1277416254421696, \"learning_rate\": 0.00018211492091337042}\r\n",
      "{\"epoch\": 59, \"train_loss\": 0.15579482161931227, \"val_loss\": 0.11946972388681024, \"learning_rate\": 0.00018151277957285543}\r\n",
      "{\"epoch\": 60, \"train_loss\": 0.1544896195717156, \"val_loss\": 0.13800262578576802, \"learning_rate\": 0.00018090169943749476}\r\n",
      "{\"epoch\": 61, \"train_loss\": 0.15482090326733888, \"val_loss\": 0.16262787098856643, \"learning_rate\": 0.00018028174751911146}\r\n",
      "{\"epoch\": 62, \"train_loss\": 0.15214620767964515, \"val_loss\": 0.11730648927995935, \"learning_rate\": 0.00017965299180241963}\r\n",
      "{\"epoch\": 63, \"train_loss\": 0.15184011068870312, \"val_loss\": 0.17353092773724346, \"learning_rate\": 0.000179015501237569}\r\n",
      "{\"epoch\": 64, \"train_loss\": 0.15085316296236123, \"val_loss\": 0.12290896245650947, \"learning_rate\": 0.00017836934573258397}\r\n",
      "{\"epoch\": 65, \"train_loss\": 0.1481425711759366, \"val_loss\": 0.12598654041811824, \"learning_rate\": 0.00017771459614569708}\r\n",
      "{\"epoch\": 66, \"train_loss\": 0.15322932218266652, \"val_loss\": 0.14818341863900422, \"learning_rate\": 0.00017705132427757892}\r\n",
      "{\"epoch\": 67, \"train_loss\": 0.1466210788594093, \"val_loss\": 0.1351458707237616, \"learning_rate\": 0.00017637960286346423}\r\n",
      "{\"epoch\": 68, \"train_loss\": 0.15004513500588945, \"val_loss\": 0.1962813674537465, \"learning_rate\": 0.00017569950556517563}\r\n",
      "{\"epoch\": 69, \"train_loss\": 0.15266709203393547, \"val_loss\": 0.12888043002318592, \"learning_rate\": 0.00017501110696304596}\r\n",
      "{\"epoch\": 70, \"train_loss\": 0.14674076704808978, \"val_loss\": 0.11937463142513298, \"learning_rate\": 0.00017431448254773944}\r\n",
      "{\"epoch\": 71, \"train_loss\": 0.15135184372058139, \"val_loss\": 0.12283471122384071, \"learning_rate\": 0.00017360970871197344}\r\n",
      "{\"epoch\": 72, \"train_loss\": 0.14754552882208954, \"val_loss\": 0.1779169113226235, \"learning_rate\": 0.00017289686274214118}\r\n",
      "{\"epoch\": 73, \"train_loss\": 0.14930408375586848, \"val_loss\": 0.1251910596564412, \"learning_rate\": 0.00017217602280983623}\r\n",
      "{\"epoch\": 74, \"train_loss\": 0.14607053263217676, \"val_loss\": 0.16195916049880907, \"learning_rate\": 0.00017144726796328034}\r\n",
      "{\"epoch\": 75, \"train_loss\": 0.15106997044305318, \"val_loss\": 0.13325791952386498, \"learning_rate\": 0.00017071067811865476}\r\n",
      "{\"epoch\": 76, \"train_loss\": 0.14775097574286628, \"val_loss\": 0.15554355247633067, \"learning_rate\": 0.00016996633405133658}\r\n",
      "{\"epoch\": 77, \"train_loss\": 0.14638480342820986, \"val_loss\": 0.14286539229238407, \"learning_rate\": 0.0001692143173870407}\r\n",
      "{\"epoch\": 78, \"train_loss\": 0.1461257400496863, \"val_loss\": 0.13688652191869916, \"learning_rate\": 0.0001684547105928689}\r\n",
      "{\"epoch\": 79, \"train_loss\": 0.14521474925547373, \"val_loss\": 0.11530959493573754, \"learning_rate\": 0.0001676875969682661}\r\n",
      "{\"epoch\": 80, \"train_loss\": 0.14332591733848676, \"val_loss\": 0.11383275331556797, \"learning_rate\": 0.00016691306063588586}\r\n",
      "{\"epoch\": 81, \"train_loss\": 0.14362086118562148, \"val_loss\": 0.11396480331127531, \"learning_rate\": 0.00016613118653236524}\r\n",
      "{\"epoch\": 82, \"train_loss\": 0.14589886908619665, \"val_loss\": 0.12128957817330957, \"learning_rate\": 0.00016534206039901063}\r\n",
      "{\"epoch\": 83, \"train_loss\": 0.14605888861545827, \"val_loss\": 0.158857210427057, \"learning_rate\": 0.00016454576877239513}\r\n",
      "{\"epoch\": 84, \"train_loss\": 0.14390414967320395, \"val_loss\": 0.12980391670949756, \"learning_rate\": 0.00016374239897486904}\r\n",
      "{\"epoch\": 85, \"train_loss\": 0.14585230090052356, \"val_loss\": 0.13269216578640045, \"learning_rate\": 0.0001629320391049838}\r\n",
      "{\"epoch\": 86, \"train_loss\": 0.14134927069847472, \"val_loss\": 0.1265235872687772, \"learning_rate\": 0.0001621147780278311}\r\n",
      "{\"epoch\": 87, \"train_loss\": 0.14185248681104276, \"val_loss\": 0.1720164961433038, \"learning_rate\": 0.00016129070536529771}\r\n",
      "{\"epoch\": 88, \"train_loss\": 0.14260108551261946, \"val_loss\": 0.11738766075577586, \"learning_rate\": 0.00016045991148623756}\r\n",
      "{\"epoch\": 89, \"train_loss\": 0.14502582531133668, \"val_loss\": 0.17097210397571325, \"learning_rate\": 0.00015962248749656164}\r\n",
      "{\"epoch\": 90, \"train_loss\": 0.14633621927923524, \"val_loss\": 0.1518099419977516, \"learning_rate\": 0.00015877852522924737}\r\n",
      "{\"epoch\": 91, \"train_loss\": 0.1453443293373799, \"val_loss\": 0.12017039644077886, \"learning_rate\": 0.00015792811723426794}\r\n",
      "{\"epoch\": 92, \"train_loss\": 0.1432104821062414, \"val_loss\": 0.14580889589083382, \"learning_rate\": 0.00015707135676844327}\r\n",
      "{\"epoch\": 93, \"train_loss\": 0.14512919970168733, \"val_loss\": 0.11412583885062486, \"learning_rate\": 0.00015620833778521315}\r\n",
      "{\"epoch\": 94, \"train_loss\": 0.14289118082033236, \"val_loss\": 0.15440066266059876, \"learning_rate\": 0.00015533915492433448}\r\n",
      "{\"epoch\": 95, \"train_loss\": 0.14384030307625653, \"val_loss\": 0.11534720625495538, \"learning_rate\": 0.00015446390350150278}\r\n",
      "{\"epoch\": 96, \"train_loss\": 0.1407218209673185, \"val_loss\": 0.12543151531182228, \"learning_rate\": 0.00015358267949789974}\r\n",
      "{\"epoch\": 97, \"train_loss\": 0.14274082440934727, \"val_loss\": 0.13544005612749607, \"learning_rate\": 0.00015269557954966786}\r\n",
      "{\"epoch\": 98, \"train_loss\": 0.1417565115855541, \"val_loss\": 0.12676003904640676, \"learning_rate\": 0.0001518027009373131}\r\n",
      "{\"epoch\": 99, \"train_loss\": 0.14529283917001448, \"val_loss\": 0.11610327929630876, \"learning_rate\": 0.00015090414157503722}\r\n",
      "{\"epoch\": 100, \"train_loss\": 0.14375351766331587, \"val_loss\": 0.1200568021866493, \"learning_rate\": 0.00015000000000000007}\r\n",
      "{\"epoch\": 101, \"train_loss\": 0.139866995933, \"val_loss\": 0.11272714131465182, \"learning_rate\": 0.00014909037536151417}\r\n",
      "{\"epoch\": 102, \"train_loss\": 0.14271209343343508, \"val_loss\": 0.11515171835571528, \"learning_rate\": 0.0001481753674101716}\r\n",
      "{\"epoch\": 103, \"train_loss\": 0.13795436160550453, \"val_loss\": 0.12134835080616176, \"learning_rate\": 0.00014725507648690549}\r\n",
      "{\"epoch\": 104, \"train_loss\": 0.14163886017577024, \"val_loss\": 0.15366331642400474, \"learning_rate\": 0.00014632960351198626}\r\n",
      "{\"epoch\": 105, \"train_loss\": 0.13838702150712487, \"val_loss\": 0.13832685987465085, \"learning_rate\": 0.00014539904997395477}\r\n",
      "{\"epoch\": 106, \"train_loss\": 0.1403272830877686, \"val_loss\": 0.11423275582678616, \"learning_rate\": 0.00014446351791849282}\r\n",
      "{\"epoch\": 107, \"train_loss\": 0.13919807779963594, \"val_loss\": 0.1507622166587971, \"learning_rate\": 0.00014352310993723285}\r\n",
      "{\"epoch\": 108, \"train_loss\": 0.13772085298553574, \"val_loss\": 0.14277785534597934, \"learning_rate\": 0.00014257792915650734}\r\n",
      "{\"epoch\": 109, \"train_loss\": 0.13797104343180544, \"val_loss\": 0.11036657778872178, \"learning_rate\": 0.0001416280792260402}\r\n",
      "{\"epoch\": 110, \"train_loss\": 0.13763981701207814, \"val_loss\": 0.1169965537097305, \"learning_rate\": 0.0001406736643075801}\r\n",
      "{\"epoch\": 111, \"train_loss\": 0.1373890013825847, \"val_loss\": 0.14035437351092697, \"learning_rate\": 0.00013971478906347814}\r\n",
      "{\"epoch\": 112, \"train_loss\": 0.13984817330101504, \"val_loss\": 0.1177083802940324, \"learning_rate\": 0.0001387515586452104}\r\n",
      "{\"epoch\": 113, \"train_loss\": 0.13813114619450062, \"val_loss\": 0.122396283862181, \"learning_rate\": 0.00013778407868184683}\r\n",
      "{\"epoch\": 114, \"train_loss\": 0.13764162183240988, \"val_loss\": 0.10934153209300712, \"learning_rate\": 0.0001368124552684679}\r\n",
      "{\"epoch\": 115, \"train_loss\": 0.13465715845199303, \"val_loss\": 0.11990893297828734, \"learning_rate\": 0.0001358367949545301}\r\n",
      "{\"epoch\": 116, \"train_loss\": 0.13846172787609975, \"val_loss\": 0.16997064832784237, \"learning_rate\": 0.00013485720473218162}\r\n",
      "{\"epoch\": 117, \"train_loss\": 0.1345503459312022, \"val_loss\": 0.13055031499592587, \"learning_rate\": 0.00013387379202452925}\r\n",
      "{\"epoch\": 118, \"train_loss\": 0.13827471094968496, \"val_loss\": 0.11037073693121784, \"learning_rate\": 0.0001328866646738584}\r\n",
      "{\"epoch\": 119, \"train_loss\": 0.13584277716246435, \"val_loss\": 0.11193580519547686, \"learning_rate\": 0.00013189593092980707}\r\n",
      "{\"epoch\": 120, \"train_loss\": 0.13923315460858868, \"val_loss\": 0.1252748105637729, \"learning_rate\": 0.00013090169943749482}\r\n",
      "{\"epoch\": 121, \"train_loss\": 0.13594603494738694, \"val_loss\": 0.1202225695680827, \"learning_rate\": 0.00012990407922560873}\r\n",
      "{\"epoch\": 122, \"train_loss\": 0.1354893821598962, \"val_loss\": 0.1370969877243042, \"learning_rate\": 0.00012890317969444724}\r\n",
      "{\"epoch\": 123, \"train_loss\": 0.13235454003054184, \"val_loss\": 0.10962916080327705, \"learning_rate\": 0.00012789911060392302}\r\n",
      "{\"epoch\": 124, \"train_loss\": 0.13784401761204937, \"val_loss\": 0.11085569503426086, \"learning_rate\": 0.00012689198206152665}\r\n",
      "{\"epoch\": 125, \"train_loss\": 0.13508913019442698, \"val_loss\": 0.13552842313051225, \"learning_rate\": 0.00012588190451025218}\r\n",
      "{\"epoch\": 126, \"train_loss\": 0.1330022666866891, \"val_loss\": 0.14241188277094624, \"learning_rate\": 0.00012486898871648554}\r\n",
      "{\"epoch\": 127, \"train_loss\": 0.13250535447692965, \"val_loss\": 0.11269906007894315, \"learning_rate\": 0.00012385334575785816}\r\n",
      "{\"epoch\": 128, \"train_loss\": 0.1340405246945098, \"val_loss\": 0.1721675228960812, \"learning_rate\": 0.00012283508701106566}\r\n",
      "{\"epoch\": 129, \"train_loss\": 0.13522603855973575, \"val_loss\": 0.1581338051678613, \"learning_rate\": 0.00012181432413965435}\r\n",
      "{\"epoch\": 130, \"train_loss\": 0.13284685482408387, \"val_loss\": 0.11874640594609082, \"learning_rate\": 0.00012079116908177601}\r\n",
      "{\"epoch\": 131, \"train_loss\": 0.1331837891074363, \"val_loss\": 0.15180385376536287, \"learning_rate\": 0.00011976573403791266}\r\n",
      "{\"epoch\": 132, \"train_loss\": 0.137756941578025, \"val_loss\": 0.12291663357091602, \"learning_rate\": 0.00011873813145857256}\r\n",
      "{\"epoch\": 133, \"train_loss\": 0.1304512494819355, \"val_loss\": 0.1122669995019678, \"learning_rate\": 0.00011770847403195841}\r\n",
      "{\"epoch\": 134, \"train_loss\": 0.1334954515836085, \"val_loss\": 0.11902217400260269, \"learning_rate\": 0.0001166768746716103}\r\n",
      "{\"epoch\": 135, \"train_loss\": 0.132967565091094, \"val_loss\": 0.11701021626777947, \"learning_rate\": 0.00011564344650402317}\r\n",
      "{\"epoch\": 136, \"train_loss\": 0.13067256973255426, \"val_loss\": 0.11834750544186681, \"learning_rate\": 0.00011460830285624125}\r\n",
      "{\"epoch\": 137, \"train_loss\": 0.1355839898491453, \"val_loss\": 0.13645057739317418, \"learning_rate\": 0.00011357155724343052}\r\n",
      "{\"epoch\": 138, \"train_loss\": 0.13458171274729538, \"val_loss\": 0.15859677687333898, \"learning_rate\": 0.0001125333233564305}\r\n",
      "{\"epoch\": 139, \"train_loss\": 0.13247503273251932, \"val_loss\": 0.1535345370322466, \"learning_rate\": 0.00011149371504928675}\r\n",
      "{\"epoch\": 140, \"train_loss\": 0.13320462897042745, \"val_loss\": 0.12486759838438592, \"learning_rate\": 0.00011045284632676544}\r\n",
      "{\"epoch\": 141, \"train_loss\": 0.13628351040667622, \"val_loss\": 0.14219307320937513, \"learning_rate\": 0.00010941083133185153}\r\n",
      "{\"epoch\": 142, \"train_loss\": 0.13140916074493433, \"val_loss\": 0.14750963826547378, \"learning_rate\": 0.00010836778433323162}\r\n",
      "{\"epoch\": 143, \"train_loss\": 0.1265654512606212, \"val_loss\": 0.12091087397933006, \"learning_rate\": 0.00010732381971276325}\r\n",
      "{\"epoch\": 144, \"train_loss\": 0.1303158122185967, \"val_loss\": 0.15880974806565792, \"learning_rate\": 0.00010627905195293142}\r\n",
      "{\"epoch\": 145, \"train_loss\": 0.12783182508700994, \"val_loss\": 0.11044312827661633, \"learning_rate\": 0.00010523359562429447}\r\n",
      "{\"epoch\": 146, \"train_loss\": 0.13045306854774244, \"val_loss\": 0.1374268567627296, \"learning_rate\": 0.00010418756537292005}\r\n",
      "{\"epoch\": 147, \"train_loss\": 0.13241644720656331, \"val_loss\": 0.11289872026361991, \"learning_rate\": 0.00010314107590781288}\r\n",
      "{\"epoch\": 148, \"train_loss\": 0.13055838474812917, \"val_loss\": 0.13827966285310686, \"learning_rate\": 0.00010209424198833577}\r\n",
      "{\"epoch\": 149, \"train_loss\": 0.13091747417364968, \"val_loss\": 0.12971945453342051, \"learning_rate\": 0.00010104717841162465}\r\n",
      "{\"epoch\": 150, \"train_loss\": 0.13441365241948516, \"val_loss\": 0.1163927743891254, \"learning_rate\": 0.00010000000000000007}\r\n",
      "{\"epoch\": 151, \"train_loss\": 0.1283366549639264, \"val_loss\": 0.18033840648899785, \"learning_rate\": 9.89528215883755e-05}\r\n",
      "{\"epoch\": 152, \"train_loss\": 0.13019136273721232, \"val_loss\": 0.13171626755967736, \"learning_rate\": 9.79057580116644e-05}\r\n",
      "{\"epoch\": 153, \"train_loss\": 0.1293513696786482, \"val_loss\": 0.11233320834068582, \"learning_rate\": 9.685892409218724e-05}\r\n",
      "{\"epoch\": 154, \"train_loss\": 0.1314420378151466, \"val_loss\": 0.11683486301917582, \"learning_rate\": 9.58124346270801e-05}\r\n",
      "Early stopping at epoch 154; best epoch was 114\r\n",
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
   "id": "5ef97fc9",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T19:16:28.653463Z",
     "iopub.status.busy": "2026-04-02T19:16:28.652804Z",
     "iopub.status.idle": "2026-04-02T19:18:06.841180Z",
     "shell.execute_reply": "2026-04-02T19:18:06.840416Z"
    },
    "papermill": {
     "duration": 98.198503,
     "end_time": "2026-04-02T19:18:06.843152",
     "exception": false,
     "start_time": "2026-04-02T19:16:28.644649",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/working/DropCount/evaluate.py:177: MatplotlibDeprecationWarning: The 'labels' parameter of boxplot() has been renamed 'tick_labels' since Matplotlib 3.9; support for the old name will be dropped in 3.11.\r\n",
      "  ax.boxplot(data, labels=[\"DL\", \"Naive\", \"MLE\"], showfliers=False)\r\n",
      "    method   count        mae       rmse    rmsle  median_relative_error  mean_relative_bias       r2  pearson  spearman  mae_log\r\n",
      "   pred_dl 10000.0  59.427266 190.977165 0.599745               0.066681            0.526563 0.997608 0.999318  0.974174 0.327304\r\n",
      "pred_naive 10000.0 356.751853 592.915548 2.400005               1.400184           39.430429 0.976940 0.997526  0.862176 1.637833\r\n",
      "  pred_mle 10000.0 264.026729 308.386464 2.400159               1.403854           39.453350 0.993762 0.999151  0.862530 1.634595\r\n"
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
   "id": "703c1e32",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-04-02T19:18:06.859956Z",
     "iopub.status.busy": "2026-04-02T19:18:06.859663Z",
     "iopub.status.idle": "2026-04-02T19:18:06.988006Z",
     "shell.execute_reply": "2026-04-02T19:18:06.987116Z"
    },
    "papermill": {
     "duration": 0.138506,
     "end_time": "2026-04-02T19:18:06.989483",
     "exception": false,
     "start_time": "2026-04-02T19:18:06.850977",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "total 171060\r\n",
      "drwxr-xr-x 3 root root      4096 Apr  2 19:16 .\r\n",
      "drwxr-xr-x 3 root root      4096 Apr  2 13:10 ..\r\n",
      "-rw-r--r-- 1 root root   3358709 Apr  2 17:42 best_model.pt\r\n",
      "-rw-r--r-- 1 root root      1268 Apr  2 19:16 config.yaml\r\n",
      "drwxr-xr-x 2 root root      4096 Apr  2 19:18 evaluation\r\n",
      "-rw-r--r-- 1 root root        78 Apr  2 19:16 summary.json\r\n",
      "-rw-r--r-- 1 root root     69161 Apr  2 19:16 training_curves.png\r\n",
      "-rw-r--r-- 1 root root     10120 Apr  2 19:16 training_history.csv\r\n",
      "-rw-r--r-- 1 root root 171703801 Apr  2 19:16 validation_dataset.pt\r\n"
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
   "duration": 22104.766261,
   "end_time": "2026-04-02T19:18:07.715082",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-04-02T13:09:42.948821",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
