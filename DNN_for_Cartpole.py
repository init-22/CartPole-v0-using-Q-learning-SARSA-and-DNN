import gym
import random
import numpy as np
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
from statistics import median, mean
from collections import Counter

LR = 1e-3
env = gym.make("CartPole-v0")
env.reset()
goal_steps = 500
score_requirement = 50
initial_games = 1000

def initial_population():
        training_data = []
        scores = []
        accepted_scores = []
        score_requirements= 50
        for _ in range(initial_games):
            observation  = env.reset()
            t = 0
            score = 0
            game_memory = []
            for i in range(200):
                t = t+1
                action  = random.randrange(0,2)
                #print(observation)
                game_memory.append([observation, action])
                observation, reward, done, info = env.step(action)
                score = score+reward
                if done:
                    print("Episode finished after {} timesteps".format(t+1))
                    break
            if score >= score_requirements:
               accepted_scores.append(score)
               
               for data in game_memory:
                   if data[1] == 1:
                        output = [0,1]
                   elif data[1] == 0:
                        output = [1,0]
                    
             
                   training_data.append([data[0], output])
               
             
      
            scores.append(score)
        env.close()
  
        training_data_save = np.array(training_data)
        np.save('saved.npy',training_data_save)
    
        # some stats here, to further illustrate the neural network magic!
        print('Average accepted score:',mean(accepted_scores))
        print('Median score for accepted scores:',median(accepted_scores))
        print(Counter(accepted_scores))
    
        return training_data    

    
def neural_network_model(input_size):

        network = input_data(shape=[None, input_size, 1], name='input')

        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 512, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 256, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 128, activation='relu')
        network = dropout(network, 0.8)

        network = fully_connected(network, 2, activation='softmax')
        network = regression(network, optimizer='adam', learning_rate=LR, loss='categorical_crossentropy', name='targets')
        model = tflearn.DNN(network, tensorboard_dir='log')

        return model    

def train_model(training_data, model=False):

        X = np.array([i[0] for i in training_data]).reshape(-1,len(training_data[0][0]),1)
        y = [i[1] for i in training_data]

        if not model:
            model = neural_network_model(input_size = len(X[0]))
        
        model.fit({'input': X}, {'targets': y}, n_epoch=5, snapshot_step=500, show_metric=True, run_id='openai_learning')
        return model    



if __name__ == "__main__":
    population = initial_population()
    model = train_model(population)
    scores = []
    choices = []
    for each_game in range(10):
        score = 0
        game_memory = []
        prev_obs = []
        env.reset()
        for _ in range(goal_steps):
            env.render()

            if len(prev_obs)==0:
                action = random.randrange(0,2)
            else:
                action = np.argmax(model.predict(prev_obs.reshape(-1,len(prev_obs),1))[0])

            choices.append(action)
                    
            new_observation, reward, done, info = env.step(action)
            prev_obs = new_observation
            game_memory.append([new_observation, action])
            score+=reward
            if done: break

        scores.append(score)

    print('Average Score:',sum(scores)/len(scores))
    print('choice 1:{}  choice 0:{}'.format(choices.count(1)/len(choices),choices.count(0)/len(choices)))
    print(score_requirement)

