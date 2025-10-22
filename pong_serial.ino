/*  Gra PONG z AI, sterowanie potencjometrem
 *  - Arduino liczy fizykę i AI
 *    Podlaczenie potencjometru B10k:
 *    skrajny prawy -> GND, skrajny lewy -> 5V , środek -> A0 - wejscie analogowe (sprawdzic przed gra)
 */

#include <Arduino.h>
#include "pong_policy.h"   

// PARAMETRY GRY
const int   W = 128, H = 64;
const int   P_W = 2, P_H = 16;
const float P_SPEED = 2.0f;            

const int   BALL_SIZE = 2;
float bx = 64, by = 32;
float bvx = 1.5f, bvy = 1.0f;

// PALETKI
const int P_AI_X = 4;
float p_ai_y = (H - P_H) / 2.0f;

const int P_PL_X = W - 4 - P_W;
float p_pl_y = (H - P_H) / 2.0f;

// PUNKTACJA
unsigned int score_ai = 0;
unsigned int score_pl = 0;

// OBSLUGA POTENCJOMETRU
const uint8_t POT_PIN = A0;
const float POT_EMA_ALPHA = 0.25f; 
float pot_filt = 0.0f;               

// TEMPO GRY
const uint16_t FRAME_MS = 15;       
unsigned long lastFrameMs = 0;

// ReLU
static inline float relu(float x){ return x > 0 ? x : 0; }

int ai_action(float bx_, float by_, float bvx_, float bvy_, float py_ai){
  const float in[NN_IN] = {
    (bx_ - 64.0f)/64.0f,
    (by_ - 32.0f)/32.0f,
    bvx_/2.0f,
    bvy_/2.0f,
    (py_ai - 32.0f)/32.0f
  };

  float h[NN_H];
  for(int i=0;i<NN_H;++i){
    float s = NN_B1[i];
    for(int j=0;j<NN_IN;++j) s += NN_W1[i][j]*in[j];
    h[i] = relu(s);
  }

  float o[NN_OUT];
  for(int k=0;k<NN_OUT;++k){
    float s = NN_B2[k];
    for(int i=0;i<NN_H;++i) s += NN_W2[k][i]*h[i];
    o[k] = s;
  }

  int a = 0; if(o[1] > o[a]) a = 1; if(o[2] > o[a]) a = 2; // 0=UP,1=STAY,2=DOWN
  return a;
}

void resetBall(int dir /* +1: w prawo, -1: w lewo */){
  bx = W/2.0f; by = H/2.0f;
  bvx = (dir>=0 ? 1.5f : -1.5f);
  bvy = (random(-10,10))/10.0f; 
}

// ODCZYT POTENCJOMETRU -> mapowanie pozycji
// WAZNE!!! -> w przypadku gdy paletka porusza sie odwrotnie, zamienic podlaczenie
//             dla skrajnych pinow potencjometru czyli GND z +5V
void readPaddleFromPot(){
  int raw = analogRead(POT_PIN);          
  // EMA (wygładzanie)
  pot_filt = POT_EMA_ALPHA * raw + (1.0f - POT_EMA_ALPHA) * pot_filt;

  // ADC 0->1023
  float mapped = (pot_filt / 1023.0f) * (H - P_H);
  p_pl_y = mapped;

  if(p_pl_y < 0) p_pl_y = 0;
  if(p_pl_y > H - P_H) p_pl_y = H - P_H;
}

// RUCH PALETKI AI
void applyAI(){
  int act = ai_action(bx, by, bvx, bvy, p_ai_y);
  if(act==0)      p_ai_y -= P_SPEED;
  else if(act==2) p_ai_y += P_SPEED;
  if(p_ai_y < 0) p_ai_y = 0;
  if(p_ai_y > H - P_H) p_ai_y = H - P_H;
}

// GLOWNA FIZYKA GRY
void physics(){
  bx += bvx; by += bvy;

  // ruch gora/dol
  if(by <= 0){ by = 0; bvy = -bvy; }
  if(by >= H - BALL_SIZE){ by = H - BALL_SIZE; bvy = -bvy; }

  // kolizja z paletka -> dla AI
  if(bvx < 0 && bx <= P_AI_X + P_W && by + BALL_SIZE >= p_ai_y && by <= p_ai_y + P_H){
    bx = P_AI_X + P_W + 1; bvx = fabs(bvx);
    float pC = p_ai_y + P_H/2.0f, bC = by + BALL_SIZE/2.0f;
    bvy += 0.6f * ((bC - pC)/(P_H/2.0f));
  }

  // kolizja z paletka -> dla gracza
  if(bvx > 0 && bx + BALL_SIZE >= P_PL_X && by + BALL_SIZE >= p_pl_y && by <= p_pl_y + P_H){
    bx = P_PL_X - BALL_SIZE - 1; bvx = -fabs(bvx);
    float pC = p_pl_y + P_H/2.0f, bC = by + BALL_SIZE/2.0f;
    bvy += 0.6f * ((bC - pC)/(P_H/2.0f));
  }

  // punkty
  if(bx < 0){        // lewo -> punkt dla gracza
    score_pl++; resetBall(-1);
  } else if(bx > W){ // prawo -> punkt dla AI
    score_ai++; resetBall(+1);
  }
}

// WYSYLANIE DANYCH DO KOMPUTERA
void sendState(){
  Serial.print("S,");
  Serial.print((int)bx); Serial.print(",");
  Serial.print((int)by); Serial.print(",");
  Serial.print((int)p_ai_y); Serial.print(",");
  Serial.print((int)p_pl_y); Serial.print(",");
  Serial.print(score_ai); Serial.print(",");
  Serial.print(score_pl); Serial.print("\n");
}

// GLOWNA PETLA i SETUP
// Ustawienie szybkosci transmisji -> testowano i 115200 daje rade
void setup(){
  Serial.begin(115200);
  randomSeed(analogRead(A0));     
  pot_filt = analogRead(POT_PIN);
  resetBall(+1);                  
}

void loop(){
  unsigned long now = millis();
  if(now - lastFrameMs < FRAME_MS) return;
  lastFrameMs = now;

  readPaddleFromPot(); 
  applyAI();
  physics();
  sendState();
}
