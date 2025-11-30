import ast
import copy
import logging
import argparse
import os
import random
import pandas as pd
import torch
import sys
sys.path.append("..")
import architectures.constrained_time as arch
import utilities.scheduling as scheduling
import matplotlib.pyplot as plt
import numpy as np
from differentiation.differentiate import first_derivative, second_derivative

logging.basicConfig(level=logging.INFO, format='%(asctime)-15s %(levelname)s %(message)s')

def main(
    outputDirectory,
    randomSeed,
    initialPosition,
    initialSpeed,
    architecture,
    startingNeuralNetwork,
    duration,
    k,
    mass,
    gamma,
    scheduleFilepath,
    numberOfDiffEquResPoints,
    displayResults
):
    device = 'cpu'
    if torch.cuda.is_available():
        device = 'cuda'

    logging.info(f"train.main(); device = {device}; architecture = {architecture}")

    outputDirectory += "_" + architecture
    if not os.path.exists(outputDirectory):
        os.makedirs(outputDirectory)

    random.seed(randomSeed)
    torch.manual_seed(randomSeed)

    # Create the neural network
    neural_net = None
    architecture_tokens = architecture.split('_')
    if architecture_tokens[0] == 'HardConstrainedTimeResNet':
        neural_net = arch.HardConstrainedTimeResNet(
            number_of_blocks=int(architecture_tokens[1]),
            block_width=int(architecture_tokens[2]),
            number_of_outputs=int(architecture_tokens[3]),
            initial_position=initialPosition,
            initial_speed=initialSpeed * duration,
            time_bubble_sigma=float(architecture_tokens[4]),
            device=device
            )
    else:
        raise NotImplementedError(f"train.main(): Not implemented architecture '{architecture}'")
    if startingNeuralNetwork is not None:
        neural_net.load_state_dict(torch.load(startingNeuralNetwork))
    neural_net.to(device)

    # Load the schedule
    schedule_df = pd.read_csv(scheduleFilepath)
    schedule = scheduling.Schedule(schedule_df)

    # Training parameters
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(neural_net.parameters(), lr=schedule.parameters(1)['learning_rate'], weight_decay=0)

    # Training loop
    minimum_loss = float('inf')
    champion_neural_net = None
    phase = schedule.parameters(1)['phase']
    diff_eqn_residual_t_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 1), requires_grad=True).to(
        device)  # (N_res, 1), normalized dimensions
    duration_alpha = np.clip(schedule.parameters(1)['duration_alpha'], 0, 1.0)
    diff_eqn_residual_t_tsr[:, 0] = duration_alpha * diff_eqn_residual_t_tsr[:, 0]

    with open(os.path.join(outputDirectory, "epochLoss.csv"), 'w', buffering=1) as epoch_loss_file:
        epoch_loss_file.write("epoch,loss,is_champion\n")
        for epoch in range(1, schedule.last_epoch() + 1):
            # Set the neural network to training mode
            neural_net.train()
            neural_net.zero_grad()

            current_parameters = schedule.parameters(epoch)
            if current_parameters['phase'] != phase:
                phase = current_parameters['phase']
                logging.info(f" ---- Phase {phase} ----")
                optimizer = torch.optim.Adam(neural_net.parameters(), lr=current_parameters['learning_rate'],
                                             weight_decay=0)
                duration_alpha = np.clip(current_parameters['duration_alpha'], 0, 1.0)
                diff_eqn_residual_t_tsr = 1.0 * torch.rand((numberOfDiffEquResPoints, 1), requires_grad=True).to(
                    device)  # (N_res, 1), normalized dimensions
                diff_eqn_residual_t_tsr[:, 0] = duration_alpha * diff_eqn_residual_t_tsr[:, 0]
                minimum_loss = float('inf')

            # Differential equation residual loss
            diff_eqn_residual_t_tsr.detach_()
            diff_eqn_residual_t_tsr.requires_grad = True

            u = neural_net(diff_eqn_residual_t_tsr).squeeze()  # (N_res,)
            du_dt = first_derivative(neural_net, diff_eqn_residual_t_tsr).squeeze()  # (N_res,)
            d2u_dt2 = second_derivative(neural_net, diff_eqn_residual_t_tsr, 0)  # (N_res, 1)
            d2u_dt2 = d2u_dt2[:, 0]  # (N_res)

            diff_eqn_residual = k * u + gamma/duration * du_dt + mass/(duration**2) * d2u_dt2  # (N_res)
            diff_eqn_residual_loss = criterion(diff_eqn_residual, torch.zeros_like(diff_eqn_residual))

            loss = diff_eqn_residual_loss
            is_champion = False
            if loss.item() < minimum_loss:
                minimum_loss = loss.item()
                champion_neural_net = copy.deepcopy(neural_net)
                is_champion = True
                champion_filepath = os.path.join(outputDirectory, f"{architecture}.pth")
                torch.save(champion_neural_net.state_dict(), champion_filepath)

            logging.info(f"Epoch {epoch}: loss = {loss.item()}")
            if is_champion:
                logging.info(f" **** Champion! ****")
            epoch_loss_file.write(f"{epoch},{loss.item()},{is_champion}\n")

            loss.backward()
            optimizer.step()


    logging.info(f"minimum_loss = {minimum_loss}")

    if displayResults:
        sys.path.append("../../mass_spring/src/mass_spring")
        from simulation import MassSpring

        analytical_mass_spring = MassSpring(mass=mass, gamma=gamma, k=k, x0=initialPosition, v0=initialSpeed, zero_threshold=1e-6)
        champion_neural_net.eval()
        u = np.zeros((256,), dtype=float)  # (256,)
        analytical_u = np.zeros((256,), dtype=float)  # (256,)
        delta_T = 1.0/(256 - 1)
        for t_ndx in range(256):
            t = t_ndx * delta_T  # [0 ... 1.0]
            t_tsr = torch.tensor([[t]]).to(device)  # (1,)
            u_t = champion_neural_net(t_tsr)  # (1,)
            u[t_ndx] = u_t.cpu().detach().item()
            analytical_u[t_ndx] = analytical_mass_spring.evaluate(duration * t)

        fig, ax = plt.subplots()
        ax.plot(duration * np.linspace(0, 1, 256), analytical_u, color='green', label='analytical', linewidth=3)
        ax.plot(duration * np.linspace(0, 1, 256), u, color='red', label='prediction', linewidth=1)

        ax.legend()
        ax.set_xlabel('t')
        ax.set_ylabel('u')
        ax.grid(True)

        image_filepath = os.path.join(outputDirectory, f"prediction_{t_ndx}.png")
        plt.savefig(image_filepath)
        plt.close()  # Prevents the figure from being displayed later


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outputDirectory', help="The output directory. Default: './output_train'", default='./output_train')
    parser.add_argument('--randomSeed', help="The random seed. Default: 0", type=int, default=0)
    parser.add_argument('--initialPosition', help="The initial position. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--initialSpeed', help="The initial speed. Default: 1.0", type=float, default=1.0)
    parser.add_argument('--architecture', help="The neural network architecture. Default: 'HardConstrainedTimeResNet_2_32_1_0.3'", default='HardConstrainedTimeResNet_2_32_1_0.3')
    parser.add_argument('--startingNeuralNetwork', help="The filepath to the starting neural network. Default: 'None'", default='None')
    parser.add_argument('--duration', help="The simulation duration, in seconds. Default: 4.0", type=float, default=4.0)
    parser.add_argument('--k', help="The spring constant. in N/m. Default: 3.9478", type=float, default=3.9478)
    parser.add_argument('--mass', help="The mass. Default: 0.1", type=float, default=0.1)
    parser.add_argument('--gamma', help="The friction coefficient, in kg/s. Default: 0.2", type=float, default=0.2)
    parser.add_argument('--scheduleFilepath', help="The filepath to the training schedule. Default: './schedule.csv'", default='./schedule.csv')
    parser.add_argument('--numberOfDiffEquResPoints', help="The number of points for the differential equation residual. Default: 32768", type=int, default=32768)
    parser.add_argument('--displayResults', help="Display the results", action='store_true')
    args = parser.parse_args()
    if args.startingNeuralNetwork.upper() == 'NONE':
        args.startingNeuralNetwork = None

    main(
        args.outputDirectory,
        args.randomSeed,
        args.initialPosition,
        args.initialSpeed,
        args.architecture,
        args.startingNeuralNetwork,
        args.duration,
        args.k,
        args.mass,
        args.gamma,
        args.scheduleFilepath,
        args.numberOfDiffEquResPoints,
        args.displayResults
    )