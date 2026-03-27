{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c2fbc355",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:17.769619Z",
     "iopub.status.busy": "2026-03-27T07:56:17.768979Z",
     "iopub.status.idle": "2026-03-27T07:56:18.861578Z",
     "shell.execute_reply": "2026-03-27T07:56:18.860601Z"
    },
    "papermill": {
     "duration": 1.099451,
     "end_time": "2026-03-27T07:56:18.863579",
     "exception": false,
     "start_time": "2026-03-27T07:56:17.764128",
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
   "id": "7ac8a616",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:18.869252Z",
     "iopub.status.busy": "2026-03-27T07:56:18.868669Z",
     "iopub.status.idle": "2026-03-27T07:56:18.876210Z",
     "shell.execute_reply": "2026-03-27T07:56:18.875570Z"
    },
    "papermill": {
     "duration": 0.012245,
     "end_time": "2026-03-27T07:56:18.877768",
     "exception": false,
     "start_time": "2026-03-27T07:56:18.865523",
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
   "id": "303f0806",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:18.882853Z",
     "iopub.status.busy": "2026-03-27T07:56:18.882226Z",
     "iopub.status.idle": "2026-03-27T07:56:19.283844Z",
     "shell.execute_reply": "2026-03-27T07:56:19.282723Z"
    },
    "papermill": {
     "duration": 0.406139,
     "end_time": "2026-03-27T07:56:19.285784",
     "exception": false,
     "start_time": "2026-03-27T07:56:18.879645",
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
   "id": "4c951772",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:19.291794Z",
     "iopub.status.busy": "2026-03-27T07:56:19.291097Z",
     "iopub.status.idle": "2026-03-27T07:56:25.083883Z",
     "shell.execute_reply": "2026-03-27T07:56:25.083033Z"
    },
    "papermill": {
     "duration": 5.798188,
     "end_time": "2026-03-27T07:56:25.086029",
     "exception": false,
     "start_time": "2026-03-27T07:56:19.287841",
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
   "id": "f26c3978",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:25.092367Z",
     "iopub.status.busy": "2026-03-27T07:56:25.091772Z",
     "iopub.status.idle": "2026-03-27T07:56:32.114271Z",
     "shell.execute_reply": "2026-03-27T07:56:32.113444Z"
    },
    "papermill": {
     "duration": 7.027749,
     "end_time": "2026-03-27T07:56:32.115883",
     "exception": false,
     "start_time": "2026-03-27T07:56:25.088134",
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
   "id": "67bdb268",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:32.121392Z",
     "iopub.status.busy": "2026-03-27T07:56:32.120998Z",
     "iopub.status.idle": "2026-03-27T07:56:32.134957Z",
     "shell.execute_reply": "2026-03-27T07:56:32.134424Z"
    },
    "papermill": {
     "duration": 0.01848,
     "end_time": "2026-03-27T07:56:32.136470",
     "exception": false,
     "start_time": "2026-03-27T07:56:32.117990",
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
    "  false_positive_rate_range: [0.0, 0.01]\n",
    "\n",
    "  train_samples_per_epoch: 10000\n",
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
    "  hidden_dim: 96\n",
    "  latent_dim: 96\n",
    "  num_latents: 16\n",
    "  num_heads: 4\n",
    "  num_self_attn_layers: 2\n",
    "  dropout: 0.1\n",
    "  fourier_features: 2\n",
    "  use_rmsnorm: false\n",
    "\n",
    "training:\n",
    "  batch_size: 1\n",
    "  epochs: 200\n",
    "  learning_rate: 3.0e-4\n",
    "  weight_decay: 1.0e-4\n",
    "  grad_clip_norm: 1.0\n",
    "  loss_name: huber_log\n",
    "  linear_loss_weight: 0.05\n",
    "  huber_delta: 0.5\n",
    "  early_stopping_patience: 20\n",
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
   "id": "77208c9c",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T07:56:32.141696Z",
     "iopub.status.busy": "2026-03-27T07:56:32.141452Z",
     "iopub.status.idle": "2026-03-27T09:56:59.578168Z",
     "shell.execute_reply": "2026-03-27T09:56:59.577314Z"
    },
    "papermill": {
     "duration": 7227.441634,
     "end_time": "2026-03-27T09:56:59.580205",
     "exception": false,
     "start_time": "2026-03-27T07:56:32.138571",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\"epoch\": 1, \"train_loss\": 0.24621945560007366, \"val_loss\": 0.12872577800229193, \"learning_rate\": 0.0002999814948722491}\r\n",
      "{\"epoch\": 2, \"train_loss\": 0.12354029289006421, \"val_loss\": 0.052156036444335765, \"learning_rate\": 0.0002999259840548597}\r\n",
      "{\"epoch\": 3, \"train_loss\": 0.09914120580843896, \"val_loss\": 0.031961990610691826, \"learning_rate\": 0.0002998334812442955}\r\n",
      "{\"epoch\": 4, \"train_loss\": 0.08519255800436255, \"val_loss\": 0.053015495498060776, \"learning_rate\": 0.0002997040092642407}\r\n",
      "{\"epoch\": 5, \"train_loss\": 0.07954406137025685, \"val_loss\": 0.005463407858585697, \"learning_rate\": 0.00029953760005996916}\r\n",
      "{\"epoch\": 6, \"train_loss\": 0.07318237727352996, \"val_loss\": 0.025359310150379316, \"learning_rate\": 0.000299334294690462}\r\n",
      "{\"epoch\": 7, \"train_loss\": 0.06924921861610624, \"val_loss\": 0.02107022662195368, \"learning_rate\": 0.00029909414331827697}\r\n",
      "{\"epoch\": 8, \"train_loss\": 0.06771356426347781, \"val_loss\": 0.008974439093835827, \"learning_rate\": 0.0002988172051971717}\r\n",
      "{\"epoch\": 9, \"train_loss\": 0.06497342946828334, \"val_loss\": 0.02264279085117505, \"learning_rate\": 0.00029850354865748363}\r\n",
      "{\"epoch\": 10, \"train_loss\": 0.06489986029525462, \"val_loss\": 0.011892606267011373, \"learning_rate\": 0.0002981532510892707}\r\n",
      "{\"epoch\": 11, \"train_loss\": 0.061841465487082976, \"val_loss\": 0.006816314366356437, \"learning_rate\": 0.0002977663989232161}\r\n",
      "{\"epoch\": 12, \"train_loss\": 0.06002949632275094, \"val_loss\": 0.019574477147503787, \"learning_rate\": 0.0002973430876093033}\r\n",
      "{\"epoch\": 13, \"train_loss\": 0.06038042856469972, \"val_loss\": 0.012898520106643559, \"learning_rate\": 0.00029688342159326487}\r\n",
      "{\"epoch\": 14, \"train_loss\": 0.058171499949278345, \"val_loss\": 0.003480071711202072, \"learning_rate\": 0.00029638751429081215}\r\n",
      "{\"epoch\": 15, \"train_loss\": 0.05823259466607078, \"val_loss\": 0.01659769842156129, \"learning_rate\": 0.0002958554880596515}\r\n",
      "{\"epoch\": 16, \"train_loss\": 0.056592870272732146, \"val_loss\": 0.037251576736627615, \"learning_rate\": 0.00029528747416929463}\r\n",
      "{\"epoch\": 17, \"train_loss\": 0.05534494608525725, \"val_loss\": 0.0030903593656439058, \"learning_rate\": 0.0002946836127686697}\r\n",
      "{\"epoch\": 18, \"train_loss\": 0.054516394208656414, \"val_loss\": 0.014712645966242917, \"learning_rate\": 0.0002940440528515414}\r\n",
      "{\"epoch\": 19, \"train_loss\": 0.05317318993063118, \"val_loss\": 0.01149555055651581, \"learning_rate\": 0.00029336895221974946}\r\n",
      "{\"epoch\": 20, \"train_loss\": 0.05111328115160467, \"val_loss\": 0.020420972139189416, \"learning_rate\": 0.000292658477444273}\r\n",
      "{\"epoch\": 21, \"train_loss\": 0.05083704063066829, \"val_loss\": 0.003139733094906546, \"learning_rate\": 0.00029191280382413173}\r\n",
      "{\"epoch\": 22, \"train_loss\": 0.04944158976978692, \"val_loss\": 0.0051618715851318485, \"learning_rate\": 0.00029113211534313374}\r\n",
      "{\"epoch\": 23, \"train_loss\": 0.0488190609263776, \"val_loss\": 0.008173599835100048, \"learning_rate\": 0.00029031660462448003}\r\n",
      "{\"epoch\": 24, \"train_loss\": 0.0477000064393686, \"val_loss\": 0.01238764232466201, \"learning_rate\": 0.0002894664728832376}\r\n",
      "{\"epoch\": 25, \"train_loss\": 0.04670780065130091, \"val_loss\": 0.008725545717599061, \"learning_rate\": 0.0002885819298766929}\r\n",
      "{\"epoch\": 26, \"train_loss\": 0.04738580900388219, \"val_loss\": 0.015192021170427325, \"learning_rate\": 0.000287663193852597}\r\n",
      "{\"epoch\": 27, \"train_loss\": 0.045419332321661975, \"val_loss\": 0.007070675503186067, \"learning_rate\": 0.00028671049149531664}\r\n",
      "{\"epoch\": 28, \"train_loss\": 0.045389260285512815, \"val_loss\": 0.024547118173213676, \"learning_rate\": 0.0002857240578699028}\r\n",
      "{\"epoch\": 29, \"train_loss\": 0.042058915664313794, \"val_loss\": 0.0024779126348298632, \"learning_rate\": 0.00028470413636409215}\r\n",
      "{\"epoch\": 30, \"train_loss\": 0.042597508708107315, \"val_loss\": 0.008368568434415692, \"learning_rate\": 0.00028365097862825497}\r\n",
      "{\"epoch\": 31, \"train_loss\": 0.04150756363889193, \"val_loss\": 0.01063059546343704, \"learning_rate\": 0.00028256484451330387}\r\n",
      "{\"epoch\": 32, \"train_loss\": 0.041388656455929186, \"val_loss\": 0.005855472979236538, \"learning_rate\": 0.00028144600200657934}\r\n",
      "{\"epoch\": 33, \"train_loss\": 0.040750333153595907, \"val_loss\": 0.005408550691510527, \"learning_rate\": 0.0002802947271657285}\r\n",
      "{\"epoch\": 34, \"train_loss\": 0.038342372718144085, \"val_loss\": 0.004827112020691857, \"learning_rate\": 0.00027911130405059136}\r\n",
      "{\"epoch\": 35, \"train_loss\": 0.039067334243125564, \"val_loss\": 0.008351789681681111, \"learning_rate\": 0.00027789602465311367}\r\n",
      "{\"epoch\": 36, \"train_loss\": 0.038003095499070334, \"val_loss\": 0.02472398919995203, \"learning_rate\": 0.0002766491888253021}\r\n",
      "{\"epoch\": 37, \"train_loss\": 0.03736900199435572, \"val_loss\": 0.007590657924830339, \"learning_rate\": 0.00027537110420524036}\r\n",
      "{\"epoch\": 38, \"train_loss\": 0.03667357693375994, \"val_loss\": 0.008538744436458728, \"learning_rate\": 0.0002740620861411841}\r\n",
      "{\"epoch\": 39, \"train_loss\": 0.03593185181978763, \"val_loss\": 0.0056147919094367464, \"learning_rate\": 0.00027272245761375335}\r\n",
      "{\"epoch\": 40, \"train_loss\": 0.03505151057579348, \"val_loss\": 0.00867591033453391, \"learning_rate\": 0.00027135254915624195}\r\n",
      "{\"epoch\": 41, \"train_loss\": 0.034533259707857186, \"val_loss\": 0.005810806754045188, \"learning_rate\": 0.00026995269877306345}\r\n",
      "{\"epoch\": 42, \"train_loss\": 0.03329597399859063, \"val_loss\": 0.0141703778711817, \"learning_rate\": 0.0002685232518563534}\r\n",
      "{\"epoch\": 43, \"train_loss\": 0.03306402725878537, \"val_loss\": 0.016274649748271714, \"learning_rate\": 0.0002670645611007493}\r\n",
      "{\"epoch\": 44, \"train_loss\": 0.03209743687223743, \"val_loss\": 0.0067616765550365015, \"learning_rate\": 0.00026557698641636824}\r\n",
      "{\"epoch\": 45, \"train_loss\": 0.03210324331069081, \"val_loss\": 0.0031887565187429063, \"learning_rate\": 0.0002640608948400045}\r\n",
      "{\"epoch\": 46, \"train_loss\": 0.03151387516312908, \"val_loss\": 0.008115101614181185, \"learning_rate\": 0.0002625166604445688}\r\n",
      "{\"epoch\": 47, \"train_loss\": 0.030711646507879532, \"val_loss\": 0.0034573564126458224, \"learning_rate\": 0.0002609446642467913}\r\n",
      "{\"epoch\": 48, \"train_loss\": 0.030213023933191512, \"val_loss\": 0.0030521838359959474, \"learning_rate\": 0.00025934529411321156}\r\n",
      "{\"epoch\": 49, \"train_loss\": 0.028986185790829268, \"val_loss\": 0.004385841676293694, \"learning_rate\": 0.00025771894466447814}\r\n",
      "Early stopping at epoch 49; best epoch was 29\r\n",
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
   "id": "d1767d2d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T09:56:59.589710Z",
     "iopub.status.busy": "2026-03-27T09:56:59.589393Z",
     "iopub.status.idle": "2026-03-27T09:59:32.981313Z",
     "shell.execute_reply": "2026-03-27T09:59:32.980285Z"
    },
    "papermill": {
     "duration": 153.39885,
     "end_time": "2026-03-27T09:59:32.983054",
     "exception": false,
     "start_time": "2026-03-27T09:56:59.584204",
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
      "   pred_dl 10000.0  318.768752  500.452722 0.055581               0.020393           -0.021005 0.992398 0.999196  0.999749 0.029771\r\n",
      "pred_naive 10000.0 1250.639150 1691.234923 0.138102               0.098967           -0.077514 0.913184 0.998050  0.999745 0.108472\r\n",
      "  pred_mle 10000.0  103.038034  139.135111 0.073246               0.010059            0.021711 0.999412 0.999761  0.999776 0.020600\r\n"
     ]
    }
   ],
   "source": [
    "!python evaluate.py --config configs/one_setting_10k.yaml --run-dir outputs/one_setting_10k"
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
   "duration": 7399.662097,
   "end_time": "2026-03-27T09:59:33.806813",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-27T07:56:14.144716",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
