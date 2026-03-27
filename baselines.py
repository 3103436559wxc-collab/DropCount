{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "7d925936",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:05.498000Z",
     "iopub.status.busy": "2026-03-27T10:21:05.497313Z",
     "iopub.status.idle": "2026-03-27T10:21:06.261155Z",
     "shell.execute_reply": "2026-03-27T10:21:06.260419Z"
    },
    "papermill": {
     "duration": 0.77026,
     "end_time": "2026-03-27T10:21:06.263091",
     "exception": false,
     "start_time": "2026-03-27T10:21:05.492831",
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
      "Receiving objects: 100% (47/47), 29.96 KiB | 1.76 MiB/s, done.\r\n",
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
   "id": "610807ef",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:06.269068Z",
     "iopub.status.busy": "2026-03-27T10:21:06.268590Z",
     "iopub.status.idle": "2026-03-27T10:21:06.274773Z",
     "shell.execute_reply": "2026-03-27T10:21:06.274047Z"
    },
    "papermill": {
     "duration": 0.010797,
     "end_time": "2026-03-27T10:21:06.276258",
     "exception": false,
     "start_time": "2026-03-27T10:21:06.265461",
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
   "id": "3402df8b",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:06.281084Z",
     "iopub.status.busy": "2026-03-27T10:21:06.280779Z",
     "iopub.status.idle": "2026-03-27T10:21:06.668829Z",
     "shell.execute_reply": "2026-03-27T10:21:06.668035Z"
    },
    "papermill": {
     "duration": 0.392366,
     "end_time": "2026-03-27T10:21:06.670458",
     "exception": false,
     "start_time": "2026-03-27T10:21:06.278092",
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
   "id": "9891e5b0",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:06.675707Z",
     "iopub.status.busy": "2026-03-27T10:21:06.675461Z",
     "iopub.status.idle": "2026-03-27T10:21:11.375922Z",
     "shell.execute_reply": "2026-03-27T10:21:11.374870Z"
    },
    "papermill": {
     "duration": 4.705414,
     "end_time": "2026-03-27T10:21:11.377945",
     "exception": false,
     "start_time": "2026-03-27T10:21:06.672531",
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
   "id": "9fa3f7e6",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:11.383780Z",
     "iopub.status.busy": "2026-03-27T10:21:11.383195Z",
     "iopub.status.idle": "2026-03-27T10:21:15.664374Z",
     "shell.execute_reply": "2026-03-27T10:21:15.663408Z"
    },
    "papermill": {
     "duration": 4.285859,
     "end_time": "2026-03-27T10:21:15.665987",
     "exception": false,
     "start_time": "2026-03-27T10:21:11.380128",
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
   "id": "ad1fb237",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:15.671762Z",
     "iopub.status.busy": "2026-03-27T10:21:15.671152Z",
     "iopub.status.idle": "2026-03-27T10:21:15.685011Z",
     "shell.execute_reply": "2026-03-27T10:21:15.684121Z"
    },
    "papermill": {
     "duration": 0.01857,
     "end_time": "2026-03-27T10:21:15.686763",
     "exception": false,
     "start_time": "2026-03-27T10:21:15.668193",
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
    "  early_stopping_patience: 30\n",
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
   "id": "3bc51967",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T10:21:15.692653Z",
     "iopub.status.busy": "2026-03-27T10:21:15.691937Z",
     "iopub.status.idle": "2026-03-27T12:29:56.055246Z",
     "shell.execute_reply": "2026-03-27T12:29:56.054393Z"
    },
    "papermill": {
     "duration": 7720.368486,
     "end_time": "2026-03-27T12:29:56.057596",
     "exception": false,
     "start_time": "2026-03-27T10:21:15.689110",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "{\"epoch\": 1, \"train_loss\": 0.33746138857434627, \"val_loss\": 0.14192788049206137, \"learning_rate\": 0.0002999814948722491}\r\n",
      "{\"epoch\": 2, \"train_loss\": 0.1438311955497296, \"val_loss\": 0.03086712558370084, \"learning_rate\": 0.0002999259840548597}\r\n",
      "{\"epoch\": 3, \"train_loss\": 0.1024121312904531, \"val_loss\": 0.025237140816843747, \"learning_rate\": 0.0002998334812442955}\r\n",
      "{\"epoch\": 4, \"train_loss\": 0.08955574805956021, \"val_loss\": 0.023778987509251236, \"learning_rate\": 0.0002997040092642407}\r\n",
      "{\"epoch\": 5, \"train_loss\": 0.08755288739765496, \"val_loss\": 0.02321205659832049, \"learning_rate\": 0.00029953760005996916}\r\n",
      "{\"epoch\": 6, \"train_loss\": 0.08029256865586286, \"val_loss\": 0.022583463596442924, \"learning_rate\": 0.000299334294690462}\r\n",
      "{\"epoch\": 7, \"train_loss\": 0.0757591568817866, \"val_loss\": 0.03515547963633435, \"learning_rate\": 0.00029909414331827697}\r\n",
      "{\"epoch\": 8, \"train_loss\": 0.07357936789985142, \"val_loss\": 0.026022636713667453, \"learning_rate\": 0.0002988172051971717}\r\n",
      "{\"epoch\": 9, \"train_loss\": 0.07132074779518094, \"val_loss\": 0.017800467945751734, \"learning_rate\": 0.00029850354865748363}\r\n",
      "{\"epoch\": 10, \"train_loss\": 0.07548501505952222, \"val_loss\": 0.012067893391971665, \"learning_rate\": 0.0002981532510892707}\r\n",
      "{\"epoch\": 11, \"train_loss\": 0.07094066026839486, \"val_loss\": 0.027209213158115746, \"learning_rate\": 0.0002977663989232161}\r\n",
      "{\"epoch\": 12, \"train_loss\": 0.06573847240535555, \"val_loss\": 0.019142454009588163, \"learning_rate\": 0.0002973430876093033}\r\n",
      "{\"epoch\": 13, \"train_loss\": 0.07008254487228675, \"val_loss\": 0.02683254797204563, \"learning_rate\": 0.00029688342159326487}\r\n",
      "{\"epoch\": 14, \"train_loss\": 0.06445276524384444, \"val_loss\": 0.010913470882056714, \"learning_rate\": 0.00029638751429081215}\r\n",
      "{\"epoch\": 15, \"train_loss\": 0.06549325144315317, \"val_loss\": 0.01980672570250317, \"learning_rate\": 0.0002958554880596515}\r\n",
      "{\"epoch\": 16, \"train_loss\": 0.06311559460649795, \"val_loss\": 0.027255105930613355, \"learning_rate\": 0.00029528747416929463}\r\n",
      "{\"epoch\": 17, \"train_loss\": 0.06070279080406549, \"val_loss\": 0.01040402147862187, \"learning_rate\": 0.0002946836127686697}\r\n",
      "{\"epoch\": 18, \"train_loss\": 0.05982031776320664, \"val_loss\": 0.030489855052592246, \"learning_rate\": 0.0002940440528515414}\r\n",
      "{\"epoch\": 19, \"train_loss\": 0.05976567519387695, \"val_loss\": 0.011862496619597209, \"learning_rate\": 0.00029336895221974946}\r\n",
      "{\"epoch\": 20, \"train_loss\": 0.05728290700228113, \"val_loss\": 0.00959599771572705, \"learning_rate\": 0.000292658477444273}\r\n",
      "{\"epoch\": 21, \"train_loss\": 0.056117307029886876, \"val_loss\": 0.023549664073017992, \"learning_rate\": 0.00029191280382413173}\r\n",
      "{\"epoch\": 22, \"train_loss\": 0.05427249848657774, \"val_loss\": 0.006252495714270935, \"learning_rate\": 0.00029113211534313374}\r\n",
      "{\"epoch\": 23, \"train_loss\": 0.05563326234512689, \"val_loss\": 0.013114170312450369, \"learning_rate\": 0.00029031660462448003}\r\n",
      "{\"epoch\": 24, \"train_loss\": 0.053567493036165766, \"val_loss\": 0.02083552291468186, \"learning_rate\": 0.0002894664728832376}\r\n",
      "{\"epoch\": 25, \"train_loss\": 0.05404530746929132, \"val_loss\": 0.00790772034648495, \"learning_rate\": 0.0002885819298766929}\r\n",
      "{\"epoch\": 26, \"train_loss\": 0.05478102929758061, \"val_loss\": 0.013979378941294272, \"learning_rate\": 0.000287663193852597}\r\n",
      "{\"epoch\": 27, \"train_loss\": 0.05133563500464742, \"val_loss\": 0.023754494071006774, \"learning_rate\": 0.00028671049149531664}\r\n",
      "{\"epoch\": 28, \"train_loss\": 0.05291799279521938, \"val_loss\": 0.008412241276008217, \"learning_rate\": 0.0002857240578699028}\r\n",
      "{\"epoch\": 29, \"train_loss\": 0.047911851899477234, \"val_loss\": 0.020120453391291447, \"learning_rate\": 0.00028470413636409215}\r\n",
      "{\"epoch\": 30, \"train_loss\": 0.049445452648325325, \"val_loss\": 0.012381551232168476, \"learning_rate\": 0.00028365097862825497}\r\n",
      "{\"epoch\": 31, \"train_loss\": 0.04739411128735249, \"val_loss\": 0.038398224620483236, \"learning_rate\": 0.00028256484451330387}\r\n",
      "{\"epoch\": 32, \"train_loss\": 0.04665284212952865, \"val_loss\": 0.019519129372143652, \"learning_rate\": 0.00028144600200657934}\r\n",
      "{\"epoch\": 33, \"train_loss\": 0.046505418432462184, \"val_loss\": 0.009147486966334781, \"learning_rate\": 0.0002802947271657285}\r\n",
      "{\"epoch\": 34, \"train_loss\": 0.043617701615035594, \"val_loss\": 0.01408882910986705, \"learning_rate\": 0.00027911130405059136}\r\n",
      "{\"epoch\": 35, \"train_loss\": 0.045167095468156046, \"val_loss\": 0.026725154193389243, \"learning_rate\": 0.00027789602465311367}\r\n",
      "{\"epoch\": 36, \"train_loss\": 0.044664544905416714, \"val_loss\": 0.03800775475706905, \"learning_rate\": 0.0002766491888253021}\r\n",
      "{\"epoch\": 37, \"train_loss\": 0.043293972471594236, \"val_loss\": 0.017956286926957547, \"learning_rate\": 0.00027537110420524036}\r\n",
      "{\"epoch\": 38, \"train_loss\": 0.04260113528300169, \"val_loss\": 0.01771521578476677, \"learning_rate\": 0.0002740620861411841}\r\n",
      "{\"epoch\": 39, \"train_loss\": 0.04268994566834684, \"val_loss\": 0.02192111788284683, \"learning_rate\": 0.00027272245761375335}\r\n",
      "{\"epoch\": 40, \"train_loss\": 0.040391131814841365, \"val_loss\": 0.01778414744057227, \"learning_rate\": 0.00027135254915624195}\r\n",
      "{\"epoch\": 41, \"train_loss\": 0.03978388789673895, \"val_loss\": 0.00854336105821676, \"learning_rate\": 0.00026995269877306345}\r\n",
      "{\"epoch\": 42, \"train_loss\": 0.039024799268321794, \"val_loss\": 0.0194699739132775, \"learning_rate\": 0.0002685232518563534}\r\n",
      "{\"epoch\": 43, \"train_loss\": 0.039825587010020515, \"val_loss\": 0.02249192389687414, \"learning_rate\": 0.0002670645611007493}\r\n",
      "{\"epoch\": 44, \"train_loss\": 0.038578861391979254, \"val_loss\": 0.015486523974970624, \"learning_rate\": 0.00026557698641636824}\r\n",
      "{\"epoch\": 45, \"train_loss\": 0.038796273765815195, \"val_loss\": 0.007140564888119116, \"learning_rate\": 0.0002640608948400045}\r\n",
      "{\"epoch\": 46, \"train_loss\": 0.03738707741665946, \"val_loss\": 0.012940556198911508, \"learning_rate\": 0.0002625166604445688}\r\n",
      "{\"epoch\": 47, \"train_loss\": 0.03573625811948927, \"val_loss\": 0.028092358181756254, \"learning_rate\": 0.0002609446642467913}\r\n",
      "{\"epoch\": 48, \"train_loss\": 0.03477761503021037, \"val_loss\": 0.00972659949878107, \"learning_rate\": 0.00025934529411321156}\r\n",
      "{\"epoch\": 49, \"train_loss\": 0.034846801008129674, \"val_loss\": 0.008098017297056004, \"learning_rate\": 0.00025771894466447814}\r\n",
      "{\"epoch\": 50, \"train_loss\": 0.03472501753159354, \"val_loss\": 0.01090372584666784, \"learning_rate\": 0.0002560660171779819}\r\n",
      "{\"epoch\": 51, \"train_loss\": 0.03379399928221191, \"val_loss\": 0.018501178980236545, \"learning_rate\": 0.00025438691948884693}\r\n",
      "{\"epoch\": 52, \"train_loss\": 0.03208140844491593, \"val_loss\": 0.013004326664467441, \"learning_rate\": 0.00025268206588930313}\r\n",
      "Early stopping at epoch 52; best epoch was 22\r\n",
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
   "id": "17ddc60d",
   "metadata": {
    "execution": {
     "iopub.execute_input": "2026-03-27T12:29:56.067741Z",
     "iopub.status.busy": "2026-03-27T12:29:56.067404Z",
     "iopub.status.idle": "2026-03-27T12:32:29.301800Z",
     "shell.execute_reply": "2026-03-27T12:32:29.300650Z"
    },
    "papermill": {
     "duration": 153.241857,
     "end_time": "2026-03-27T12:32:29.303722",
     "exception": false,
     "start_time": "2026-03-27T12:29:56.061865",
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
      "   pred_dl 10000.0  317.994343  418.845686 0.115597               0.030956            0.037995 0.994675 0.997676  0.999309 0.041669\r\n",
      "pred_naive 10000.0 1135.628494 1555.501756 0.213160               0.098997            0.006747 0.926560 0.997521  0.999234 0.123988\r\n",
      "  pred_mle 10000.0  304.992047  362.578950 0.190800               0.031796            0.108735 0.996010 0.999328  0.999355 0.069607\r\n"
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
   "duration": 7887.362855,
   "end_time": "2026-03-27T12:32:30.127708",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2026-03-27T10:21:02.764853",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
