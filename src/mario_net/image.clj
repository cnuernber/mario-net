(ns mario-net.image
  (:require [mikera.image.core :as imagez]
            [clojure.java.io :as io]
            [clojure.core.matrix :as m]
            [mikera.vectorz.core]
            [cortex.description :as desc]
            [cortex.layers :as layers]
            [cortex.network :as net]
            [cortex.optimise :as opt]
            [cortex.core :as core]
            [cortex.protocols :as cp]
            [clojure.core.matrix.macros :refer [c-for]])
  (:import [java.awt.image BufferedImage]))



(defn load-main-image [] (imagez/load-image (io/resource "1-1.png")))

(defonce main-image (memoize load-main-image))


(defn get-pixel-indexes
  [^BufferedImage img]
  (let [^ints pixels (imagez/get-pixels img)
        data-set (set (seq pixels))]
    (vec (map-indexed vector data-set))))

(defn diff-squared
  ^double [lhs rhs]
  (let [tmp (double (- lhs rhs))]
    (* tmp tmp)))


(defn image-to-indexed-column-major-matrix
  [^BufferedImage img]
  (let [pixel-indexes (get-pixel-indexes img)
        pixel-to-index-map (into {} (map (comp vec reverse) pixel-indexes))
        ^ints pixels (imagez/get-pixels img)
        indexes (mapv #(pixel-to-index-map %) pixels)
        mean (double (/ (m/esum indexes) (count indexes)))
        deviation (double (/ (reduce + (map #(diff-squared mean %) indexes))
                          (count indexes)))
        indexes (map #(double (/ (- % mean) deviation)) indexes)
        width (.getWidth img)
        height (.getHeight img)]
    {:mean mean :deviation deviation
     :pixel-indexes pixel-indexes
     :data (m/pack (m/transpose (m/array :vectorz (mapv vec (partition width indexes)))))}))


(defn matrix-to-input-vectors
  [mat row-width]
  (let [last-row (- (m/row-count mat) row-width)
        mat-as-vec (m/as-vector mat)
        column-stride (m/column-count mat)
        row-width-stride (* row-width column-stride)]
    (mapv (fn [idx]
            (m/pack
             (m/subvector mat-as-vec (* idx column-stride) row-width-stride)))
          (range last-row))))


(defn image-to-mean-stddev-and-input-vectors
  [^BufferedImage img]
  (let [{:keys [mean deviation data] :as matrix-data}
        (image-to-indexed-column-major-matrix img)
        input-vecs (matrix-to-input-vectors data 5)]
    (assoc matrix-data :input-vectors input-vecs)))

(defn clamp
  [val min-val max-val]
  (max (min val max-val) min-val))


(defn mat-to-image
  [matrix {:keys [mean deviation pixel-indexes]}]
  (let [num-indexes (count pixel-indexes)
        index-map (into {} pixel-indexes)
        new-mat (m/transpose matrix)
        pixels (int-array (map (fn [val]
                                 (index-map (int (clamp (Math/round
                                                         (+ mean (* val deviation)))
                                                        0
                                                        (- num-indexes 1)))))
                               (m/eseq new-mat)))
        retval (imagez/new-image (m/column-count new-mat) (m/row-count new-mat))]
    (imagez/set-pixels retval pixels)
    retval))


(def autoencoder-layer-sizes [256 128 64 32 16])


(defn train-autoencoder-layer
  [input layer-size]
  (let [input-size (m/ecount (first input))
        network (desc/build-and-create-network [(desc/input input-size)
                                                (desc/gaussian-noise)
                                                (desc/linear->logistic layer-size)
                                                (desc/linear input-size)])
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        loss-fn (opt/mse-loss)
        network (net/train-until-error-stabilizes
                 network optimizer loss-fn
                 input input 5 input input)
        ;;drop the noise and the decoder
        min-network (assoc network :modules (vec (drop 1 (drop-last (:modules network)))))
        next-data (net/run min-network input)
        modules (:modules network)]
    {:encoder {:weights (:weights (second modules)) :bias (:bias (second modules))}
     :decoder {:weights (:weights (last  modules)) :bias (:bias (last  modules))}
     :next-input next-data}))


(defn layers->full-network
  [layers input-size with-noise]
  (let [network-input (if with-noise
                        [(desc/input input-size)
                         (desc/gaussian-noise)]
                        [(desc/input input-size)])]
   (desc/build-and-create-network
    (concat network-input
            (map (fn [layer]
                   (let [weights (get-in layer [:encoder :weights])
                         bias (get-in layer [:encoder :bias])
                         n-input (m/column-count weights)
                         n-output (m/row-count weights)]
                     (desc/linear->logistic n-output
                                            :weights weights :bias bias)))
                 layers)
            (map (fn [layer]
                   (let [weights (get-in layer [:decoder :weights])
                         bias (get-in layer [:decoder :bias])
                         n-output (m/row-count weights)]
                     (desc/linear n-output :weights weights :bias bias)))
                 (reverse layers))))))

(defn train-full-encoder-decoder-network
  [input layers]
  (let [input-size (m/ecount (first input))
        loss-fn (opt/mse-loss)
        network (layers->full-network layers input-size true)
        optimizer (opt/adadelta-optimiser (core/parameter-count network))
        network (net/train-until-error-stabilizes
                 network optimizer loss-fn
                 input input 5 input input)
        front-half (* 2 (count layers))
        ;;drop the noise layer
        modules (drop 1 (:modules network))
        encoder-layers (mapcat drop-last
                               (partition
                                2
                                (take front-half modules)))
        decoder-layers (reverse (drop front-half modules))]
    (mapv (fn [encoder decoder]
            {:encoder {:weights (:weights encoder) :bias (:bias encoder)}
             :decoder {:weights (:weights decoder) :bias (:bias decoder)}})
          encoder-layers decoder-layers)))


(def stacked-network (atom nil))
(def full-network (atom nil))

(defn train-autoencoder-network
  []
  (let [scaled-image (imagez/scale (main-image) 0.25)
        data (image-to-mean-stddev-and-input-vectors scaled-image)
        layers (reduce (fn [layers layer-size]
                         (let [input-data (or (:next-input (last layers))
                                              (:input-vectors data))]
                           (conj layers (train-autoencoder-layer input-data layer-size))))
                       []
                       autoencoder-layer-sizes)
        _ (reset! stacked-network layers)
        layers (train-full-encoder-decoder-network (:input-vectors data) layers)]
    (assoc data :layers layers :scaled-image scaled-image)))


;;render the right-most column, copy into the data store and render the next
;;column
(defn render-image-from-autoencoder-network
  [{:keys [mean variance data layers scaled-image pixel-indexes] :as data-set}]
  (let [row-width 5
        mat data
        column-stride (m/column-count mat)
        row-width-stride (* row-width column-stride)
        data-vector (m/new-array :vectorz (* row-width column-stride))
        initial-data-vector (m/subvector data-vector 0 (* (dec row-width) column-stride))
        _ (m/assign! initial-data-vector
                     (m/subvector (m/as-vector data) 0 (* (dec row-width))))
        output-mat (m/new-array :vectors (m/shape data))
        _ (m/assign! (m/subvector output-mat 0 (* (dec row-width) column-stride))
                     (m/subvector (m/as-vector data) 0 (* (dec row-width))))
        network (layers->full-network layers false)
        columns-to-render (- (m/row-count data) row-width)]
    ;;generate columns
    (doseq [idx (range columns-to-render)]
      (let [network (cp/forward network data-vector)
            output (cp/output network)]
        (m/assign! initial-data-vector
                   (m/subvector output column-stride (* (dec row-width) column-stride)))
        (m/assign! (m/subvector (m/as-vector output-mat)
                                (* (+ idx (dec row-width)) column-stride)
                                column-stride))))
    (mat-to-image output-mat data-set)))


(defn image-to-indexed-squares
  [^BufferedImage src-image ^long square-size]
  (let [input-width (.getWidth src-image)
        input-height (.getHeight src-image)
        output-width (quot input-width square-size)
        output-height (quot input-height square-size)
        ^ints input-data (imagez/get-pixels src-image)
        output-pixel-count (* output-width output-height)
        retval (reduce (fn [[index-map output-pixels] ^long output-idx]
                         (let [output-y (quot output-idx output-width)
                               output-x (rem output-idx output-width)
                               input-y (* output-y square-size)
                               input-x (* output-x square-size)
                               input-values (vec (for [y (range square-size)
                                                       x (range square-size)]
                                                   (let [input-offset (+ (* (+ input-y y)
                                                                            input-width)
                                                                         (+ input-x x))]
                                                     (aget input-data input-offset))))
                               item-idx (get index-map input-values (count index-map))]
                           [(assoc index-map input-values item-idx)
                            (conj output-pixels item-idx)]))
                       [{} []]
                       (range output-pixel-count))
        [item-map output-pixels] retval
        output-pixels (m/array :vectorz (mapv vec (partition output-width output-pixels)))]
    {:item-map item-map :pixels output-pixels}))


(defn indexed-squares-to-image
  [{:keys [item-map pixels]}]
  (let [input-square-count (count (ffirst item-map))
        square-size (int (Math/sqrt input-square-count))
        reverse-map (into {} (map (comp vec reverse) item-map))
        output-pixel-count (long (m/ecount pixels))
        input-pixel-count (* output-pixel-count square-size square-size)
        input-width (long (* square-size (m/column-count pixels)))
        retval (imagez/new-image input-width
                                 (* square-size (m/row-count pixels)))
        ^ints input-pixels (imagez/get-pixels retval)
        output-width (long (m/column-count pixels))
        output-pixel-vec (m/as-vector pixels)]
    (c-for
     [output-pixel 0 (< output-pixel output-pixel-count) (inc output-pixel)]
     (let [output-x (rem output-pixel output-width)
           output-y (quot output-pixel output-width)
           input-x (* output-x square-size)
           input-y (* output-y square-size)
           output-pixel (int (m/mget output-pixel-vec output-pixel))
           input-data (reverse-map output-pixel)]
       (c-for
        [y 0 (< y square-size) (inc y)]
        (c-for
         [x 0 (< x square-size) (inc x)]
         (let [input-val (input-data (+ (* y square-size) x))
               input-addr (+ (* (+ input-y y) input-width) (+ input-x x))]
           (aset input-pixels input-addr input-val))))))
    (imagez/set-pixels retval input-pixels)
    retval))
