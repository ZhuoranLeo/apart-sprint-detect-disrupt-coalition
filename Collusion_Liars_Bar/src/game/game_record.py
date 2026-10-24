from dataclasses import dataclass, field
from typing import List, Dict, Optional
import datetime
import json
import os

def generate_game_id():
    """Generate game ID with time information"""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    return timestamp

@dataclass
class PlayerInitialState:
    """Records the player's initial state, including pistol status and hand cards"""
    player_name: str
    bullet_position: int
    current_gun_position: int
    initial_hand: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "bullet_position": self.bullet_position,
            "current_gun_position": self.current_gun_position,
            "initial_hand": self.initial_hand
        }

@dataclass
class PlayAction:
    """Record a single act of playing a card"""
    player_name: str
    played_cards: List[str]
    remaining_cards: List[str]
    play_reason: str
    behavior: str
    next_player: str
    was_challenged: bool = False
    challenge_reason: Optional[str] = None
    challenge_result: Optional[bool] = None
    play_thinking: Optional[str] = None
    challenge_thinking: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "player_name": self.player_name,
            "played_cards": self.played_cards,
            "remaining_cards": self.remaining_cards,
            "play_reason": self.play_reason,
            "behavior": self.behavior,
            "next_player": self.next_player,
            "was_challenged": self.was_challenged,
            "challenge_reason": self.challenge_reason,
            "challenge_result": self.challenge_result,
            "play_thinking": self.play_thinking,
            "challenge_thinking": self.challenge_thinking
        }
    
    def update_challenge(self, was_challenged: bool, reason: str, result: bool, challenge_thinking: str = None) -> None:
        """Update the challenge information"""
        self.was_challenged = was_challenged
        self.challenge_reason = reason
        self.challenge_result = result
        self.challenge_thinking = challenge_thinking

@dataclass
class ShootingResult:
    """Record the result of a shooting"""
    shooter_name: str
    bullet_hit: bool
    
    def to_dict(self) -> Dict:
        return {
            "shooter_name": self.shooter_name,
            "bullet_hit": self.bullet_hit,
        }

@dataclass
class RoundRecord:
    """Record the information of a single round"""
    round_id: int
    target_card: str
    starting_player: str
    player_initial_states: List[PlayerInitialState]
    round_players: List[str] = field(default_factory=list)
    player_opinions: Dict[str, Dict[str, str]] = field(default_factory=dict)
    player_scores: Dict[str, int] = field(default_factory=dict)
    play_history: List[PlayAction] = field(default_factory=list)
    round_result: Optional[ShootingResult] = None
    
    def to_dict(self) -> Dict:
        return {
            "round_id": self.round_id,
            "target_card": self.target_card,
            "round_players": self.round_players,
            "starting_player": self.starting_player,
            "player_initial_states": [ps.to_dict() for ps in self.player_initial_states],
            "player_opinions": self.player_opinions,
            "player_scores": self.player_scores,
            "play_history": [play.to_dict() for play in self.play_history],
            "round_result": self.round_result.to_dict() if self.round_result else None
        }
    
    def add_play_action(self, action: PlayAction) -> None:
        """Add a play action to the play history"""
        self.play_history.append(action)
    
    def get_last_action(self) -> Optional[PlayAction]:
        """Get the last play action"""
        return self.play_history[-1] if self.play_history else None
    
    def set_shooting_result(self, result: ShootingResult) -> None:
        """Set the shooting result"""
        self.round_result = result

    def get_latest_round_info(self) -> str:
        """Return the basic information of the latest round"""
        return (
            f"It's round {self.round_id}, target card: {self.target_card}, player of this round：{'、'.join(self.round_players)}，"
            f"Starting from player {self.starting_player}"
        )

    def get_latest_round_actions(self, current_player: str, include_latest: bool = True) -> str:
        """
        Enter the current player and return the action information for the round
        
        Args:
            current_player (str): Current player name
            include_latest (bool): Whether to include the latest action, default is True
        
        Returns:
            str: Formatted action information
        """
        action_texts = []
        actions_to_process = self.play_history if include_latest else self.play_history[:-1]
        
        for action in actions_to_process:
            if action.player_name == current_player:
                action_texts.append(
                    f"It's your turn to play, and you play {len(action.played_cards)}cards，played cards：{'、'.join(action.played_cards)}，"
                    f"Remaining cards：{'、'.join(action.remaining_cards)}\nYour behaviour：{action.behavior}"
                )
            else:
                action_texts.append(
                    f"It's {action.player_name}'s turn to play a card， {action.player_name}claimed to have played{len(action.played_cards)}card(s) of'{self.target_card}'，"
                    f"Remaining cards：{len(action.remaining_cards)}cards, \n{action.player_name}'s behaviour：{action.behavior}"
                )
            
            if action.was_challenged:
                actual_cards = f"The card that was played was:{'、'.join(action.played_cards)}"
                challenge_result_text = f"{actual_cards}，challenge successful" if action.challenge_result else f"{actual_cards}，challenge failed"
                if action.next_player == current_player:
                    challenge_text = f"You chose to challenge{action.player_name}，{action.player_name}{challenge_result_text}"
                elif action.player_name == current_player:
                    challenge_text = f"{action.next_player}chose to challenge you，you{challenge_result_text}"
                else:
                    challenge_text = f"{action.next_player}chose to challenge{action.player_name}，{action.player_name}{challenge_result_text}"
            else:
                if action.next_player == current_player:
                    challenge_text = f"You chose to not challenge{action.player_name}"
                elif action.player_name == current_player:
                    challenge_text = f"{action.next_player}chose to not challenge you"
                else:
                    challenge_text = f"{action.next_player}chose to not challenge{action.player_name}"
            action_texts.append(challenge_text)
        
        return "\n".join(action_texts)
    
    def get_latest_play_behavior(self) -> str:
        """
        Get the latest player behaviour in the latest round
        
        Returns:
            str: Formatted player behaviour
        """
        if not self.play_history:
            return ""
            
        last_action = self.get_last_action()
        if not last_action:
            return ""
            
        return (f"{last_action.player_name}claimed to have played{len(last_action.played_cards)}card(s) of'{self.target_card}'，"
                f"remaining cards:{len(last_action.remaining_cards)}cards，"
                f"{last_action.player_name}'s behavior：{last_action.behavior}")
    
    def get_latest_round_result(self, current_player: str) -> str:
        """
        Return the latest shooting result
        
        Args:
            current_player (str): Current player name
            
        Returns:
            str: Formatted shooting result
        """
        if not self.round_result:
            return None
            
        if self.round_result.shooter_name == "No one":
            return "No one shot this round"
            
        shooter = "You" if self.round_result.shooter_name == current_player else self.round_result.shooter_name
        
        if self.round_result.bullet_hit:
            return f"{shooter}fired! Bullet hit. {shooter} is dead."
        else:
            return f"{shooter}fired！Bullet didn't hit，{shooter} is survived."

    def get_play_decision_info(self, self_player: str, interacting_player: str) -> str:
        """Get information about the current play's decisions
        
        Args:
            self_player: Current player
            interacting_player: Next player
        Returns:
            str: Information containing the gun status of both players and the current player's impression of the next player
        """
        self_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == self_player), None)
        other_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == interacting_player), None)
        opinion = self.player_opinions[self_player].get(interacting_player, "Don't know well about this player yet")
        
        return (f"{interacting_player}is the next player，who will decide to challenge your play or not.\n"
                f"You have fired{self_gun}shot(s)，{interacting_player}fired{other_gun}shot(s)."
                f"Your impression of{interacting_player}：{opinion}")

    def get_challenge_decision_info(self, self_player: str, interacting_player: str) -> str:
        """Getting information about the current round of challenge decisions
        
        Args:
            self_player: Current player
            interacting_player: Previous player
        Returns:
            str: Information containing the gun status of both players and the current player's impression of the previous player
        """
        self_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == self_player), None)
        other_gun = next((ps.current_gun_position for ps in self.player_initial_states if ps.player_name == interacting_player), None)
        opinion = self.player_opinions[self_player].get(interacting_player, "Don't know well about this player yet")
        
        return (f"You are deciding whether to challenge {interacting_player}'s claim.\n"
                f"You have fired{self_gun}shot(s)，{interacting_player}fired{other_gun}shot(s)."
                f"Your impression of{interacting_player}：{opinion}")

@dataclass
class GameRecord:
    """Full game records"""
    def __init__(self, player_configs: Optional[List[Dict[str, str]]] = None, communication_enabled: bool = False):
        self.game_id: str = generate_game_id()
        self.player_names: List[str] = []
        self.rounds: List[RoundRecord] = []
        self.winner: Optional[str] = None
        self.final_scores: Dict[str, int] = {}
        
        # Communication-related fields
        self.communication_enabled: bool = communication_enabled
        self.communication_history: List[Dict] = []
        self.communication_rounds: int = 1  # Default value
        
        # Secret channel fields
        self.secret_channel_config: Dict = {}
        self.secret_channel_events: List[Dict] = []
        self.secret_communications: List[Dict] = []
        
        # Create folder structure based on player configuration
        if player_configs:
            config_folder_name = "-".join([config["name"] for config in player_configs])
            # Add communication suffix to distinguish communication-enabled games
            if communication_enabled:
                config_folder_name += "_communication"
            self.save_directory = os.path.join("experiments", "game_records", config_folder_name)
        else:
            self.save_directory = os.path.join("experiments", "game_records")
        
        # Make sure the save directory exists
        if not os.path.exists(self.save_directory):
            os.makedirs(self.save_directory)
    
    def to_dict(self) -> Dict:
        result = {
            "game_id": self.game_id,
            "player_names": self.player_names,
            "rounds": [round.to_dict() for round in self.rounds],
            "winner": self.winner,
            "final_scores": self.final_scores
        }
        
        # Add communication fields if communication is enabled
        if self.communication_enabled:
            result.update({
                "communication_enabled": self.communication_enabled,
                "communication_history": self.communication_history,
                "communication_rounds": self.communication_rounds
            })
        
        # Experiment metadata (condition, alliance pair, seats, models)
        if getattr(self, "meta", None):
            result["meta"] = self.meta

        # Save secret-channel data in every auto-save, not only at the end of the game
        if self.secret_channel_events or self.secret_communications:
            result["secret_channel"] = {
                "events": self.secret_channel_events,
                "communications": self.secret_communications
            }
        
        return result
    
    def start_game(self, player_names: List[str]) -> None:
        """Initialize game, record player information"""
        self.player_names = player_names
        self.final_scores = {player: 0 for player in player_names}
    
    def start_round(self, round_id: int, target_card: str, round_players: List[str], 
                starting_player: str, player_initial_states: List[PlayerInitialState], 
                player_opinions: Dict[str, Dict[str, str]], player_scores: Dict[str, int] = None) -> None:
        """Starting a new round of games"""
        if player_scores is None:
            player_scores = {player: 0 for player in round_players}
        else:
            # Make a deep copy of the scores to avoid changing them retroactively
            player_scores = player_scores.copy()
                
        round_record = RoundRecord(
            round_id=round_id,
            target_card=target_card,
            round_players=round_players,
            starting_player=starting_player,
            player_initial_states=player_initial_states,
            player_opinions=player_opinions,
            player_scores=player_scores
        )
        self.rounds.append(round_record)
    
    def update_scores(self, scores: Dict[str, int]) -> None:
        """Update the scores for the current round only
        
        Args:
            scores: Dictionary of player scores
        """
        current_round = self.get_current_round()
        if current_round:
            # Only update the current round's scores, not all previous rounds
            current_round.player_scores = scores.copy()
        
        # Also update final scores for the game record
        self.final_scores = scores.copy()
    
    def record_play(self, player_name: str, played_cards: List[str], remaining_cards: List[str], play_reason: str, behavior: str, next_player: str, play_thinking: str = None) -> None:
        """Record the player's card playing behavior"""
        current_round = self.get_current_round()
        if current_round:
            play_action = PlayAction(
                player_name=player_name,
                played_cards=played_cards,
                remaining_cards=remaining_cards,
                play_reason=play_reason,
                behavior=behavior,
                next_player=next_player,
                play_thinking=play_thinking
            )
            current_round.add_play_action(play_action)
    
    def record_challenge(self, was_challenged: bool, reason: str = None, result: bool = None, challenge_thinking: str = None) -> None:
        """Record the challenge information"""
        current_round = self.get_current_round()
        if current_round:
            last_action = current_round.get_last_action()
            if last_action:
                last_action.update_challenge(was_challenged, reason, result, challenge_thinking)
    
    def record_shooting(self, shooter_name: str, bullet_hit: bool) -> None:
        """Record the shooting results"""
        current_round = self.get_current_round()
        if current_round:
            shooting_result = ShootingResult(shooter_name=shooter_name, bullet_hit=bullet_hit)
            current_round.set_shooting_result(shooting_result)
            self.auto_save()  # Save the game record after each round
    
    def finish_game(self, winner_name: str, final_scores: Dict[str, int] = None) -> None:
        """End the game and record the winner and final scores
        
        Args:
            winner_name: The name of the winner
            final_scores: Optional dictionary of final scores, if not provided uses current scores
        """
        self.winner = winner_name
        
        # Ensure final scores are recorded
        if final_scores is not None:
            self.final_scores = final_scores.copy()
        
        self.auto_save()  # Save the game record after the game ends
    
    def get_current_round(self) -> Optional[RoundRecord]:
        """Get the current round"""
        return self.rounds[-1] if self.rounds else None
    
    def get_latest_round_info(self) -> Optional[str]:
        """Get the basic information of the latest round"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_info() if current_round else None

    def get_latest_round_actions(self, current_player: str, include_latest: bool = True) -> Optional[str]:
        """Get the action information of the latest round"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_actions(current_player, include_latest) if current_round else None
    
    def get_latest_play_behavior(self) -> Optional[str]:
        """
        Get the latest player behaviour in the latest round
        """
        current_round = self.get_current_round()
        return current_round.get_latest_play_behavior() if current_round else None

    def get_latest_round_result(self, current_player: str) -> Optional[str]:
        """Get the latest shooting result"""
        current_round = self.get_current_round()
        return current_round.get_latest_round_result(current_player) if current_round else None

    def get_play_decision_info(self, self_player: str, interacting_player: str) -> Optional[str]:
        """Get information about the latest play's decisions
        """
        current_round = self.get_current_round()
        return current_round.get_play_decision_info(self_player, interacting_player) if current_round else None

    def get_challenge_decision_info(self, self_player: str, interacting_player: str) -> Optional[str]:
        """Get information about the latest round of challenge decisions
        """
        current_round = self.get_current_round()
        return current_round.get_challenge_decision_info(self_player, interacting_player) if current_round else None

    def add_communication_record(self, communication_record: Dict) -> None:
        """Add a communication record to the game history"""
        if self.communication_enabled:
            self.communication_history.append(communication_record)

    def set_communication_parameters(self, communication_rounds: int) -> None:
        """Set communication parameters for this game"""
        self.communication_rounds = communication_rounds
    
    def initialize_game(self, player_states: List[PlayerInitialState], target_card: str, 
                       first_player: str, secret_channel_config: Dict = None) -> None:
        """Initialize game record with secret channel configuration"""
        self.player_names = [ps.player_name for ps in player_states]
        self.target_card = target_card
        self.first_player = first_player
        
        # Secret channel setup
        self.secret_channel_config = secret_channel_config or {}
        self.secret_channel_events = []
        self.secret_communications = []
    
    def add_secret_channel_event(self, event: Dict) -> None:
        """Record secret channel events (offer, acceptance, establishment)"""
        self.secret_channel_events.append(event)
    
    def add_secret_communication(self, sender: str, receiver: str, message: str, 
                                response: str, round_number: int) -> None:
        """Record a secret communication between channel participants"""
        comm_record = {
            'round': round_number,
            'sender': sender,
            'receiver': receiver,
            'message': message,
            'response': response,
            'timestamp': round_number
        }
        self.secret_communications.append(comm_record)
    
    def save_game_record(self) -> None:
        """Save the complete game record including secret channel data"""
        # Add secret channel data to the record
        record_data = self.to_dict()
        if hasattr(self, 'secret_channel_config'):
            record_data['secret_channel'] = {
                'config': self.secret_channel_config,
                'events': getattr(self, 'secret_channel_events', []),
                'communications': getattr(self, 'secret_communications', [])
            }
        
        # Determine save directory based on configuration
        if hasattr(self, 'secret_channel_config') and self.secret_channel_config:
            config_folder_name = "-".join(self.player_names)
            config_folder_name += "_secret_channel"
            save_dir = os.path.join("experiments/game_records", config_folder_name)
        else:
            save_dir = self.save_directory
        
        # Ensure directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        file_path = os.path.join(save_dir, f"{self.game_id}.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(record_data, file, indent=4, ensure_ascii=False)
        print(f"Game record saved to {file_path}")

    def auto_save(self) -> None:
        """Automatically save the game record to a file"""
        file_path = os.path.join(self.save_directory, f"{self.game_id}.json")
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(self.to_dict(), file, indent=4, ensure_ascii=False)
        print(f"Game records have been save automatically to {file_path}")