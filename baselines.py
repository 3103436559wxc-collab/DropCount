{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b897503e",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:25:54.426103Z",
     "iopub.status.busy": "2026-03-29T11:25:54.425731Z",
     "iopub.status.idle": "2026-03-29T11:25:55.124601Z",
     "shell.execute_reply": "2026-03-29T11:25:55.123834Z"
    },
    "papermill": {
     "duration": 0.705517,
     "end_time": "2026-03-29T11:25:55.126529",
     "exception": false,
     "start_time": "2026-03-29T11:25:54.421012",
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
      "Receiving objects: 100% (47/47), 29.96 KiB | 5.99 MiB/s, done.\r\n",
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
   "id": "c091f7c0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:25:55.132448Z",
     "iopub.status.busy": "2026-03-29T11:25:55.131810Z",
     "iopub.status.idle": "2026-03-29T11:25:55.138506Z",
     "shell.execute_reply": "2026-03-29T11:25:55.137713Z"
    },
    "papermill": {
     "duration": 0.011219,
     "end_time": "2026-03-29T11:25:55.139993",
     "exception": false,
     "start_time": "2026-03-29T11:25:55.128774",
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
    "%cd /kaggle/working/DropCount"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "aef6f1c5",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:25:55.144591Z",
     "iopub.status.busy": "2026-03-29T11:25:55.144381Z",
     "iopub.status.idle": "2026-03-29T11:25:55.532210Z",
     "shell.execute_reply": "2026-03-29T11:25:55.531410Z"
    },
    "papermill": {
     "duration": 0.392142,
     "end_time": "2026-03-29T11:25:55.533933",
     "exception": false,
     "start_time": "2026-03-29T11:25:55.141791",
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
      "Receiving objects: 100% (47/47), 29.96 KiB | 14.98 MiB/s, done.\r\n",
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
   "execution_count": 4,
   "id": "7d3f131f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:25:55.540038Z",
     "iopub.status.busy": "2026-03-29T11:25:55.539187Z",
     "iopub.status.idle": "2026-03-29T11:26:00.117489Z",
     "shell.execute_reply": "2026-03-29T11:26:00.116740Z"
    },
    "papermill": {
     "duration": 4.583739,
     "end_time": "2026-03-29T11:26:00.119875",
     "exception": false,
     "start_time": "2026-03-29T11:25:55.536136",
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
   "execution_count": 5,
   "id": "7ca31b1f",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:26:00.126023Z",
     "iopub.status.busy": "2026-03-29T11:26:00.125217Z",
     "iopub.status.idle": "2026-03-29T11:26:04.343842Z",
     "shell.execute_reply": "2026-03-29T11:26:04.343031Z"
    },
    "papermill": {
     "duration": 4.223148,
     "end_time": "2026-03-29T11:26:04.345276",
     "exception": false,
     "start_time": "2026-03-29T11:26:00.122128",
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
   "execution_count": 6,
   "id": "48e959c2",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:26:04.351072Z",
     "iopub.status.busy": "2026-03-29T11:26:04.350526Z",
     "iopub.status.idle": "2026-03-29T11:26:04.363311Z",
     "shell.execute_reply": "2026-03-29T11:26:04.362562Z"
    },
    "papermill": {
     "duration": 0.017325,
     "end_time": "2026-03-29T11:26:04.364879",
     "exception": false,
     "start_time": "2026-03-29T11:26:04.347554",
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
    "  copy_sampling_mode: uniform_integer\n",
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
   "execution_count": 7,
   "id": "d943c505",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T11:26:04.370159Z",
     "iopub.status.busy": "2026-03-29T11:26:04.369691Z",
     "iopub.status.idle": "2026-03-29T15:48:30.181271Z",
     "shell.execute_reply": "2026-03-29T15:48:30.180134Z"
    },
    "papermill": {
     "duration": 15745.816774,
     "end_time": "2026-03-29T15:48:30.183705",
     "exception": false,
     "start_time": "2026-03-29T11:26:04.366931",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\"epoch\": 1, \"train_loss\": 0.10951900551331928, \"val_loss\": 0.04401222795248032, \"learning_rate\": 0.00019999451693655123}\r\n",
      "{\"epoch\": 2, \"train_loss\": 0.057699972079694274, \"val_loss\": 0.017355554854497312, \"learning_rate\": 0.00019997806834748456}\r\n",
      "{\"epoch\": 3, \"train_loss\": 0.04763173241724726, \"val_loss\": 0.022761366341263054, \"learning_rate\": 0.00019995065603657316}\r\n",
      "{\"epoch\": 4, \"train_loss\": 0.041831554771680386, \"val_loss\": 0.014054900139570236, \"learning_rate\": 0.00019991228300988585}\r\n",
      "{\"epoch\": 5, \"train_loss\": 0.03927831212065648, \"val_loss\": 0.013755428998731077, \"learning_rate\": 0.0001998629534754574}\r\n",
      "{\"epoch\": 6, \"train_loss\": 0.03593467326278332, \"val_loss\": 0.012768345324322581, \"learning_rate\": 0.00019980267284282717}\r\n",
      "{\"epoch\": 7, \"train_loss\": 0.035673848727031145, \"val_loss\": 0.007575138375628739, \"learning_rate\": 0.00019973144772244582}\r\n",
      "{\"epoch\": 8, \"train_loss\": 0.032247262183949355, \"val_loss\": 0.012829812711104751, \"learning_rate\": 0.00019964928592495045}\r\n",
      "{\"epoch\": 9, \"train_loss\": 0.032205218155658806, \"val_loss\": 0.054764465637505054, \"learning_rate\": 0.000199556196460308}\r\n",
      "{\"epoch\": 10, \"train_loss\": 0.031229840351943858, \"val_loss\": 0.0060878438223153355, \"learning_rate\": 0.00019945218953682734}\r\n",
      "{\"epoch\": 11, \"train_loss\": 0.029864591834857127, \"val_loss\": 0.01784925986453891, \"learning_rate\": 0.00019933727656003966}\r\n",
      "{\"epoch\": 12, \"train_loss\": 0.029817771399160847, \"val_loss\": 0.009087189434212632, \"learning_rate\": 0.00019921147013144782}\r\n",
      "{\"epoch\": 13, \"train_loss\": 0.029672292512306013, \"val_loss\": 0.006633993368363008, \"learning_rate\": 0.0001990747840471444}\r\n",
      "{\"epoch\": 14, \"train_loss\": 0.02844972165003419, \"val_loss\": 0.009085762361995877, \"learning_rate\": 0.00019892723329629887}\r\n",
      "{\"epoch\": 15, \"train_loss\": 0.028442947676451877, \"val_loss\": 0.00591315934935119, \"learning_rate\": 0.0001987688340595138}\r\n",
      "{\"epoch\": 16, \"train_loss\": 0.028663891606614925, \"val_loss\": 0.015807450875174253, \"learning_rate\": 0.0001985996037070505}\r\n",
      "{\"epoch\": 17, \"train_loss\": 0.027261506666406057, \"val_loss\": 0.003825570011162199, \"learning_rate\": 0.00019841956079692417}\r\n",
      "{\"epoch\": 18, \"train_loss\": 0.02659575179114472, \"val_loss\": 0.004462212767335586, \"learning_rate\": 0.00019822872507286888}\r\n",
      "{\"epoch\": 19, \"train_loss\": 0.02662500383538427, \"val_loss\": 0.0035430918098427354, \"learning_rate\": 0.00019802711746217218}\r\n",
      "{\"epoch\": 20, \"train_loss\": 0.026057618780515622, \"val_loss\": 0.009361055741086602, \"learning_rate\": 0.00019781476007338055}\r\n",
      "{\"epoch\": 21, \"train_loss\": 0.026217199503304436, \"val_loss\": 0.007350976573769003, \"learning_rate\": 0.0001975916761938747}\r\n",
      "{\"epoch\": 22, \"train_loss\": 0.025622850515623578, \"val_loss\": 0.02180802671983838, \"learning_rate\": 0.000197357890287316}\r\n",
      "{\"epoch\": 23, \"train_loss\": 0.025957865976088214, \"val_loss\": 0.015578767851460725, \"learning_rate\": 0.0001971134279909636}\r\n",
      "{\"epoch\": 24, \"train_loss\": 0.02484657598824706, \"val_loss\": 0.006682257722131908, \"learning_rate\": 0.00019685831611286308}\r\n",
      "{\"epoch\": 25, \"train_loss\": 0.024675454914593137, \"val_loss\": 0.009813016356900335, \"learning_rate\": 0.00019659258262890678}\r\n",
      "{\"epoch\": 26, \"train_loss\": 0.02442023280414287, \"val_loss\": 0.005410105422371999, \"learning_rate\": 0.00019631625667976578}\r\n",
      "{\"epoch\": 27, \"train_loss\": 0.024040748023986817, \"val_loss\": 0.00834445447823964, \"learning_rate\": 0.00019602936856769426}\r\n",
      "{\"epoch\": 28, \"train_loss\": 0.024005125742650124, \"val_loss\": 0.0024358026711270215, \"learning_rate\": 0.00019573194975320668}\r\n",
      "{\"epoch\": 29, \"train_loss\": 0.023057854232855605, \"val_loss\": 0.003107966153183952, \"learning_rate\": 0.00019542403285162765}\r\n",
      "{\"epoch\": 30, \"train_loss\": 0.023204772654431872, \"val_loss\": 0.006820963064674288, \"learning_rate\": 0.00019510565162951534}\r\n",
      "{\"epoch\": 31, \"train_loss\": 0.02275830973694101, \"val_loss\": 0.0068982634270796555, \"learning_rate\": 0.00019477684100095856}\r\n",
      "{\"epoch\": 32, \"train_loss\": 0.022619263161323032, \"val_loss\": 0.008354017100064084, \"learning_rate\": 0.0001944376370237481}\r\n",
      "{\"epoch\": 33, \"train_loss\": 0.022258449389087036, \"val_loss\": 0.003245179249672219, \"learning_rate\": 0.0001940880768954225}\r\n",
      "{\"epoch\": 34, \"train_loss\": 0.02213213762837695, \"val_loss\": 0.012820537092164158, \"learning_rate\": 0.0001937281989491891}\r\n",
      "{\"epoch\": 35, \"train_loss\": 0.02150470970828319, \"val_loss\": 0.011819373477250338, \"learning_rate\": 0.0001933580426497201}\r\n",
      "{\"epoch\": 36, \"train_loss\": 0.021604112619080115, \"val_loss\": 0.003265433180727996, \"learning_rate\": 0.00019297764858882508}\r\n",
      "{\"epoch\": 37, \"train_loss\": 0.021184606486774282, \"val_loss\": 0.001945414991612779, \"learning_rate\": 0.00019258705848099945}\r\n",
      "{\"epoch\": 38, \"train_loss\": 0.021091815401177154, \"val_loss\": 0.008064125278033317, \"learning_rate\": 0.00019218631515885003}\r\n",
      "{\"epoch\": 39, \"train_loss\": 0.020292324727959932, \"val_loss\": 0.010147537387907505, \"learning_rate\": 0.0001917754625683981}\r\n",
      "{\"epoch\": 40, \"train_loss\": 0.020789281774300616, \"val_loss\": 0.0044767140282783655, \"learning_rate\": 0.00019135454576426006}\r\n",
      "{\"epoch\": 41, \"train_loss\": 0.02006098630296765, \"val_loss\": 0.0027398605369962753, \"learning_rate\": 0.00019092361090470685}\r\n",
      "{\"epoch\": 42, \"train_loss\": 0.02011577402743278, \"val_loss\": 0.0021919803090859205, \"learning_rate\": 0.00019048270524660196}\r\n",
      "{\"epoch\": 43, \"train_loss\": 0.02007444719747873, \"val_loss\": 0.006762259750626981, \"learning_rate\": 0.00019003187714021935}\r\n",
      "{\"epoch\": 44, \"train_loss\": 0.01993259261827916, \"val_loss\": 0.005231357558863238, \"learning_rate\": 0.00018957117602394128}\r\n",
      "{\"epoch\": 45, \"train_loss\": 0.019943561902520016, \"val_loss\": 0.023392839808017016, \"learning_rate\": 0.00018910065241883677}\r\n",
      "{\"epoch\": 46, \"train_loss\": 0.019308835091418588, \"val_loss\": 0.008307125495746732, \"learning_rate\": 0.00018862035792312145}\r\n",
      "{\"epoch\": 47, \"train_loss\": 0.019492969574581367, \"val_loss\": 0.014082170518115163, \"learning_rate\": 0.00018813034520649919}\r\n",
      "{\"epoch\": 48, \"train_loss\": 0.019228844746854157, \"val_loss\": 0.001956965518445941, \"learning_rate\": 0.00018763066800438633}\r\n",
      "{\"epoch\": 49, \"train_loss\": 0.018990075120248365, \"val_loss\": 0.002161466332268901, \"learning_rate\": 0.00018712138111201895}\r\n",
      "{\"epoch\": 50, \"train_loss\": 0.018369727570808028, \"val_loss\": 0.002541874031565385, \"learning_rate\": 0.00018660254037844386}\r\n",
      "{\"epoch\": 51, \"train_loss\": 0.018809086153295358, \"val_loss\": 0.003778611684218049, \"learning_rate\": 0.00018607420270039436}\r\n",
      "{\"epoch\": 52, \"train_loss\": 0.018491211389086677, \"val_loss\": 0.006550135721918196, \"learning_rate\": 0.00018553642601605065}\r\n",
      "{\"epoch\": 53, \"train_loss\": 0.01809278382823104, \"val_loss\": 0.007028481684625149, \"learning_rate\": 0.0001849892692986864}\r\n",
      "{\"epoch\": 54, \"train_loss\": 0.017614164011576214, \"val_loss\": 0.002328250799968373, \"learning_rate\": 0.00018443279255020152}\r\n",
      "{\"epoch\": 55, \"train_loss\": 0.017843336633581203, \"val_loss\": 0.006408036819659174, \"learning_rate\": 0.00018386705679454242}\r\n",
      "{\"epoch\": 56, \"train_loss\": 0.017222106606158194, \"val_loss\": 0.0019648323192959652, \"learning_rate\": 0.00018329212407100997}\r\n",
      "{\"epoch\": 57, \"train_loss\": 0.017229304333857727, \"val_loss\": 0.0019799805815564468, \"learning_rate\": 0.0001827080574274562}\r\n",
      "{\"epoch\": 58, \"train_loss\": 0.01708499414799735, \"val_loss\": 0.005061509317019954, \"learning_rate\": 0.00018211492091337042}\r\n",
      "{\"epoch\": 59, \"train_loss\": 0.016819900569366292, \"val_loss\": 0.0032588364355033263, \"learning_rate\": 0.00018151277957285543}\r\n",
      "{\"epoch\": 60, \"train_loss\": 0.0168231430200045, \"val_loss\": 0.0038937541161431, \"learning_rate\": 0.00018090169943749476}\r\n",
      "{\"epoch\": 61, \"train_loss\": 0.016798610183468555, \"val_loss\": 0.004144359451485798, \"learning_rate\": 0.00018028174751911146}\r\n",
      "{\"epoch\": 62, \"train_loss\": 0.016785741506074554, \"val_loss\": 0.0039025108487694524, \"learning_rate\": 0.00017965299180241963}\r\n",
      "{\"epoch\": 63, \"train_loss\": 0.01635313549155835, \"val_loss\": 0.003595499109243974, \"learning_rate\": 0.000179015501237569}\r\n",
      "{\"epoch\": 64, \"train_loss\": 0.015832084446493536, \"val_loss\": 0.0075161775322631005, \"learning_rate\": 0.00017836934573258397}\r\n",
      "{\"epoch\": 65, \"train_loss\": 0.015836480806150938, \"val_loss\": 0.006308035934343934, \"learning_rate\": 0.00017771459614569708}\r\n",
      "{\"epoch\": 66, \"train_loss\": 0.01603242215382925, \"val_loss\": 0.002996120157884434, \"learning_rate\": 0.00017705132427757892}\r\n",
      "{\"epoch\": 67, \"train_loss\": 0.01541458126581274, \"val_loss\": 0.0032555162959033625, \"learning_rate\": 0.00017637960286346423}\r\n",
      "{\"epoch\": 68, \"train_loss\": 0.015776789327018197, \"val_loss\": 0.0056126055726781485, \"learning_rate\": 0.00017569950556517563}\r\n",
      "{\"epoch\": 69, \"train_loss\": 0.015172086712601595, \"val_loss\": 0.008750585841014982, \"learning_rate\": 0.00017501110696304596}\r\n",
      "{\"epoch\": 70, \"train_loss\": 0.015537315430224408, \"val_loss\": 0.007994046936044469, \"learning_rate\": 0.00017431448254773944}\r\n",
      "{\"epoch\": 71, \"train_loss\": 0.015134616681479383, \"val_loss\": 0.00393526538903825, \"learning_rate\": 0.00017360970871197344}\r\n",
      "{\"epoch\": 72, \"train_loss\": 0.01485731677301228, \"val_loss\": 0.0012418812978430651, \"learning_rate\": 0.00017289686274214118}\r\n",
      "{\"epoch\": 73, \"train_loss\": 0.0150294319904875, \"val_loss\": 0.0031403197397012264, \"learning_rate\": 0.00017217602280983623}\r\n",
      "{\"epoch\": 74, \"train_loss\": 0.014368730116437654, \"val_loss\": 0.001312124613323249, \"learning_rate\": 0.00017144726796328034}\r\n",
      "{\"epoch\": 75, \"train_loss\": 0.014745943428727332, \"val_loss\": 0.0035402086754329504, \"learning_rate\": 0.00017071067811865476}\r\n",
      "{\"epoch\": 76, \"train_loss\": 0.01447987717104843, \"val_loss\": 0.005649741875939071, \"learning_rate\": 0.00016996633405133658}\r\n",
      "{\"epoch\": 77, \"train_loss\": 0.014264639934341539, \"val_loss\": 0.0020782727535115556, \"learning_rate\": 0.0001692143173870407}\r\n",
      "{\"epoch\": 78, \"train_loss\": 0.013962676514079795, \"val_loss\": 0.003483945615589619, \"learning_rate\": 0.0001684547105928689}\r\n",
      "{\"epoch\": 79, \"train_loss\": 0.013833517921419116, \"val_loss\": 0.0026429846610408277, \"learning_rate\": 0.0001676875969682661}\r\n",
      "{\"epoch\": 80, \"train_loss\": 0.013653258466301487, \"val_loss\": 0.0023723553365562113, \"learning_rate\": 0.00016691306063588586}\r\n",
      "{\"epoch\": 81, \"train_loss\": 0.013900682255561696, \"val_loss\": 0.0035600697761401534, \"learning_rate\": 0.00016613118653236524}\r\n",
      "{\"epoch\": 82, \"train_loss\": 0.013967522546055261, \"val_loss\": 0.001415509833372198, \"learning_rate\": 0.00016534206039901063}\r\n",
      "{\"epoch\": 83, \"train_loss\": 0.013605476930289297, \"val_loss\": 0.0034223072854802014, \"learning_rate\": 0.00016454576877239513}\r\n",
      "{\"epoch\": 84, \"train_loss\": 0.013841953983396525, \"val_loss\": 0.007956701811403037, \"learning_rate\": 0.00016374239897486904}\r\n",
      "{\"epoch\": 85, \"train_loss\": 0.013132794219075004, \"val_loss\": 0.0017533827356237452, \"learning_rate\": 0.0001629320391049838}\r\n",
      "{\"epoch\": 86, \"train_loss\": 0.012810667940403801, \"val_loss\": 0.0054250634917989374, \"learning_rate\": 0.0001621147780278311}\r\n",
      "{\"epoch\": 87, \"train_loss\": 0.013089967674796935, \"val_loss\": 0.002256683403567877, \"learning_rate\": 0.00016129070536529771}\r\n",
      "{\"epoch\": 88, \"train_loss\": 0.01282916147178039, \"val_loss\": 0.0014714491942431779, \"learning_rate\": 0.00016045991148623756}\r\n",
      "{\"epoch\": 89, \"train_loss\": 0.012917721911647823, \"val_loss\": 0.0027103409017436206, \"learning_rate\": 0.00015962248749656164}\r\n",
      "{\"epoch\": 90, \"train_loss\": 0.012355294479615987, \"val_loss\": 0.001603410809650086, \"learning_rate\": 0.00015877852522924737}\r\n",
      "{\"epoch\": 91, \"train_loss\": 0.0126349394030869, \"val_loss\": 0.007764792411122471, \"learning_rate\": 0.00015792811723426794}\r\n",
      "{\"epoch\": 92, \"train_loss\": 0.011881819088140037, \"val_loss\": 0.002707196835370269, \"learning_rate\": 0.00015707135676844327}\r\n",
      "{\"epoch\": 93, \"train_loss\": 0.012432832884253002, \"val_loss\": 0.004310147398151457, \"learning_rate\": 0.00015620833778521315}\r\n",
      "{\"epoch\": 94, \"train_loss\": 0.012214963570405963, \"val_loss\": 0.003426470338832587, \"learning_rate\": 0.00015533915492433448}\r\n",
      "{\"epoch\": 95, \"train_loss\": 0.011732184266083641, \"val_loss\": 0.0018919149284483865, \"learning_rate\": 0.00015446390350150278}\r\n",
      "{\"epoch\": 96, \"train_loss\": 0.0121329808638955, \"val_loss\": 0.002188870426500216, \"learning_rate\": 0.00015358267949789974}\r\n",
      "{\"epoch\": 97, \"train_loss\": 0.01220085369014414, \"val_loss\": 0.002205431603360921, \"learning_rate\": 0.00015269557954966786}\r\n",
      "{\"epoch\": 98, \"train_loss\": 0.012011119529569987, \"val_loss\": 0.004519811915233731, \"learning_rate\": 0.0001518027009373131}\r\n",
      "{\"epoch\": 99, \"train_loss\": 0.011651501747290604, \"val_loss\": 0.003025145823834464, \"learning_rate\": 0.00015090414157503722}\r\n",
      "{\"epoch\": 100, \"train_loss\": 0.01142420575778233, \"val_loss\": 0.002307000709697604, \"learning_rate\": 0.00015000000000000007}\r\n",
      "{\"epoch\": 101, \"train_loss\": 0.011655323484481778, \"val_loss\": 0.0019689240182051435, \"learning_rate\": 0.00014909037536151417}\r\n",
      "{\"epoch\": 102, \"train_loss\": 0.011219028586079366, \"val_loss\": 0.004857498464174569, \"learning_rate\": 0.0001481753674101716}\r\n",
      "{\"epoch\": 103, \"train_loss\": 0.011385693587968126, \"val_loss\": 0.0018882841293234379, \"learning_rate\": 0.00014725507648690549}\r\n",
      "{\"epoch\": 104, \"train_loss\": 0.011036675598169676, \"val_loss\": 0.0018207509772619232, \"learning_rate\": 0.00014632960351198626}\r\n",
      "{\"epoch\": 105, \"train_loss\": 0.011358602252230048, \"val_loss\": 0.0013049433347187005, \"learning_rate\": 0.00014539904997395477}\r\n",
      "{\"epoch\": 106, \"train_loss\": 0.01098965537113836, \"val_loss\": 0.0022396391551010313, \"learning_rate\": 0.00014446351791849282}\r\n",
      "{\"epoch\": 107, \"train_loss\": 0.010663837891834555, \"val_loss\": 0.005789819200523198, \"learning_rate\": 0.00014352310993723285}\r\n",
      "{\"epoch\": 108, \"train_loss\": 0.010599577527854126, \"val_loss\": 0.006656819154508412, \"learning_rate\": 0.00014257792915650734}\r\n",
      "{\"epoch\": 109, \"train_loss\": 0.010685796658619073, \"val_loss\": 0.0035863723068032412, \"learning_rate\": 0.0001416280792260402}\r\n",
      "{\"epoch\": 110, \"train_loss\": 0.0106947523147217, \"val_loss\": 0.0027033592415973543, \"learning_rate\": 0.0001406736643075801}\r\n",
      "{\"epoch\": 111, \"train_loss\": 0.010269132179836743, \"val_loss\": 0.0018876399187138305, \"learning_rate\": 0.00013971478906347814}\r\n",
      "{\"epoch\": 112, \"train_loss\": 0.01041085610400769, \"val_loss\": 0.0022878755204146727, \"learning_rate\": 0.0001387515586452104}\r\n",
      "Early stopping at epoch 112; best epoch was 72\r\n",
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
   "execution_count": 8,
   "id": "e43019f3",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T15:48:30.198568Z",
     "iopub.status.busy": "2026-03-29T15:48:30.198172Z",
     "iopub.status.idle": "2026-03-29T15:50:25.886398Z",
     "shell.execute_reply": "2026-03-29T15:50:25.885709Z"
    },
    "papermill": {
     "duration": 115.697866,
     "end_time": "2026-03-29T15:50:25.888325",
     "exception": false,
     "start_time": "2026-03-29T15:48:30.190459",
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
      "    method   count         mae        rmse    rmsle  median_relative_error  mean_relative_bias       r2  pearson  spearman  mae_log\r\n",
      "   pred_dl 10000.0  117.105666  160.674675 0.047533               0.011361           -0.006507 0.999216 0.999697  0.999750 0.017704\r\n",
      "pred_naive 10000.0 1135.628494 1555.501756 0.213160               0.098997            0.006747 0.926560 0.997521  0.999234 0.123988\r\n",
      "  pred_mle 10000.0  304.992047  362.578950 0.190800               0.031796            0.108735 0.996010 0.999328  0.999355 0.069607\r\n"
     ]
    }
   ],
   "source": [
    "!python evaluate.py --config configs/one_setting_10k.yaml --run-dir outputs/one_setting_10k"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "7a20c505",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-29T15:50:25.902922Z",
     "iopub.status.busy": "2026-03-29T15:50:25.902109Z",
     "iopub.status.idle": "2026-03-29T15:50:25.919445Z",
     "shell.execute_reply": "2026-03-29T15:50:25.918907Z"
    },
    "papermill": {
     "duration": 0.026315,
     "end_time": "2026-03-29T15:50:25.920938",
     "exception": false,
     "start_time": "2026-03-29T15:50:25.894623",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/working/DropCount\n",
      "total 32\n",
      "drwxr-xr-x 8 root root 4096 Mar 29 11:25 .\n",
      "drwxr-xr-x 1 root root 4096 Mar 29 11:25 ..\n",
      "drwxr-xr-x 2 root root 4096 Mar 29 11:25 huggingface\n",
      "drwxr-xr-x 2 root root 4096 Mar 29 11:25 input\n",
      "drwxr-xr-x 3 root root 4096 Mar 29 11:25 lib\n",
      "drwxr-xr-x 2 root root 4096 Mar 29 11:25 nbdev\n",
      "drwxr-xr-x 2 root root 4096 Mar 29 11:25 src\n",
      "drwxr-xr-x 3 root root 4096 Mar 29 11:25 working\n",
      "total 68\n",
      "drwxr-xr-x 3 root root  4096 Mar 29 11:25 .\n",
      "drwxr-xr-x 8 root root  4096 Mar 29 11:25 ..\n",
      "drwxr-xr-x 8 root root  4096 Mar 29 11:26 DropCount\n",
      "---------- 1 root root 54062 Mar 29 15:50 __notebook__.ipynb\n"
     ]
    }
   ],
   "source": [
    "%%bash\n",
    "pwd\n",
    "ls -la /kaggle\n",
    "ls -la /kaggle/working"
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
   "duration": 15874.976084,
   "end_time": "2026-03-29T15:50:26.746752",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-29T11:25:51.770668",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
