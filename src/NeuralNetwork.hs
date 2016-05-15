{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE NamedFieldPuns #-}
module NeuralNetwork where

import Data.Array.Repa as Repa
import Data.Array.Repa.Algorithms.Matrix
import qualified Data.Array.Repa.Operators.Mapping as Mapping

import Control.Monad (replicateM)

import System.Random (randomRIO)

import Control.Monad.Identity (runIdentity)


type Vector a = Array a DIM2 Double
type Matrix = Array U DIM2 Double

newtype NN = NN { thetas :: [Matrix] } deriving Show

test = do
  let (nn, xys) = createSampleNN
      iter = iterate (\nn -> gradDescent 0.2 nn xys) nn
      nn' = iter !! 10000
  print nn'
  print $ o nn' (fst $ xys !! 0)
  print $ o nn' (fst $ xys !! 1)
  print $ o nn' (fst $ xys !! 2)
  print $ o nn' (fst $ xys !! 3)
  where o nn x = computeUnboxedS $ removeBias $ last $ outputs nn x

compute = runIdentity . computeUnboxedP

createRandomMatrix :: Int -> Int -> IO Matrix
createRandomMatrix x y = do
  elems <- replicateM (x*y) $ randomRIO (-1, 1)
  return $ fromListUnboxed (Z :. x :. y) elems

createSampleNN :: (NN, [(Vector U, Vector U)])
createSampleNN = (NN thetas, xys)
  where thetas :: [Matrix]
        thetas = [ fromListUnboxed (Z :. 2 :. 3) [ 0.4, -0.6, 0.5, -0.2, -0.3, 0.4]
                 , fromListUnboxed (Z :. 1 :. 3) [ 0.6, -0.5, 0.2 ]
                 ]
        xDim = Z :. 2 :. 1
        yDim = Z :. 1 :. 1
        xys = [ (fromListUnboxed xDim [ 1.0, 0.0 ],
                 fromListUnboxed yDim [ 1.0 ] )
              , (fromListUnboxed xDim [ 0.0, 1.0 ],
                 fromListUnboxed yDim [ 1.0 ])
              , (fromListUnboxed xDim [ 0.0, 0.0 ],
                 fromListUnboxed yDim [ 0.0 ])
              , (fromListUnboxed xDim [ 1.0, 1.0 ],
                 fromListUnboxed yDim [ 0.0 ])
              ]

gradDescentS :: Double -> NN -> (Vector U, Vector U) -> NN
gradDescentS alpha nn@(NN { thetas }) (x, y) = NN $ Prelude.zipWith (\x y -> compute $ x +^ y) thetas corr
  where corr = Prelude.map (Mapping.map (\v -> -alpha * v)) derivs
        derivs =
          let outs = outputs nn x in
          Prelude.zipWith (\x y -> delay $ x `mmultS` y) (tail $ deltas nn outs y) (Prelude.map transpose2S outs)


gradDescent :: Double -> NN -> [(Vector U, Vector U)] -> NN
gradDescent alpha nn@(NN { thetas }) xys = NN $ Prelude.zipWith (\x y -> compute $ x +^ y) thetas corr
  where corr = Prelude.map (Mapping.map (\v -> -alpha * v / m)) $ derivSum xys
        m = fromIntegral $ length xys
        derivSum [xy] = derivs xy
        derivSum (xy:xys) = Prelude.zipWith (\x y -> x +^ y) (derivs xy) (derivSum xys)
        derivs (x, y) =
          let outs = outputs nn x in
          Prelude.zipWith (\x y -> delay $ x `mmultS` y) (tail $ deltas nn outs y) (Prelude.map transpose2S outs)

{-# INLINE deltas #-}
deltas :: NN -> [Vector U] -> Vector U -> [Vector U]
deltas (NN { thetas }) outs y = scanr f deltaLast $ zip outs thetas
  where deltaLast = compute $ (removeBias $ last outs) -^ y
        f :: (Vector U, Matrix) -> Vector U -> Vector U
        f (out, theta) delta =
          let out' = removeBias out in
          let z = Mapping.map (\x -> x * (1 - x)) out' in
          let td = removeBias $ transpose2S theta `mmultS` delta in
          let result = compute $ td *^ z in
          result

{-# INLINE outputs #-}
outputs :: NN -> Vector U -> [Vector U]
outputs (NN { thetas }) x = scanl f (addBias x) thetas
  where f :: Vector U -> Matrix -> Vector U
        f a theta =
          addBias
          $ Mapping.map sigmoid
          $ (theta `mmultS` a)

{-# INLINE addBias #-}
addBias :: Source r Double => Vector r -> Vector U
addBias x = compute $ transpose $ Repa.append one $ transpose x
  where one :: Vector U
        one = fromListUnboxed (Z :. 1 :. 1) [ 1.0 ]

{-# INLINE removeBias #-}
removeBias :: Vector U -> Vector D
removeBias vec = extract (Z :. 1 :. 0) (Z :. (x-1) :. 1) vec
  where (Z :. x) :. 1 = extent vec

{-# INLINE sigmoid #-}
sigmoid :: Double -> Double
sigmoid x = 1 / (1 + exp (-x))
